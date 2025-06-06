import sys
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint
import gym

from karel_env.tool.syntax_checker import PySyntaxChecker
#from karel_env.karel_supervised import Karel_world_supervised
from karel_env.karel_supervised_new_vae import Karel_world_supervised

from rl.distributions import FixedCategorical, FixedNormal
from rl.model_option import NNBase
from rl.utils import masked_mean, masked_sum, create_hook, init


class Flatten(nn.Module):
    def forward(self, x):
       #return x.view(x.size(0), -1)
       return x.reshape(x.size(0), -1)

#TODO: replace _unmask_idx with _unmask_idx2 after verifying identity
def _unmask_idx(output_mask_all, first_end_token_idx, max_program_len):
    for p_idx in range(first_end_token_idx.shape[0]):
        t_idx = int(first_end_token_idx[p_idx].detach().cpu().numpy())
        if t_idx < max_program_len:
            output_mask_all[p_idx, t_idx] = True
    return output_mask_all.to(torch.bool)

def _unmask_idx2(x):
    seq, seq_len = x
    if seq_len < seq.shape[0]:
        seq[seq_len] = True
        return True
    return False


class Normal(nn.Module):
    def __init__(self):
        super(Normal, self).__init__()

    def forward(self, mean, std):
        return FixedNormal(mean, std)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

# For RoPE
# from torchtune.modules import RotaryPositionalEmbeddings

from rl.model_option import NNBase
from rl.utils import init

class ActionBehaviorEncoder(NNBase):
    def __init__(
        self,
        recurrent,
        num_actions,
        hidden_size=64,
        rnn_type='GRU',
        dropout=0.0,
        use_linear=False,
        unit_size=256,
        **kwargs
    ):
        """
        Encodes a rollout of actions (only) into a single latent vector per rollout.

        :param num_actions: how many possible actions exist (for embedding).
        :param hidden_size: final output dimension (matching ProgramEncoder).
        :param rnn_type: 'GRU' or 'LSTM'.
        :param dropout: dropout on the RNN.
        :param use_linear: if True, project from `unit_size` down to `hidden_size`.
        :param unit_size: internal dimension for the RNN and/or embedding.
        """
        # We set recurrent=True so NNBase will handle RNN init for us
        super(ActionBehaviorEncoder, self).__init__(
            recurrent=True,
            recurrent_input_size=unit_size,   # dimension fed into the RNN
            hidden_size=unit_size,  # RNN’s hidden dimension
            dropout=dropout,
            rnn_type=rnn_type
        )

        # Same style init as ConditionPolicy, etc.
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu')
        )

        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self._use_linear = use_linear
        self.unit_size = unit_size
        self.state_shape = (kwargs['input_channel'], kwargs['input_height'], kwargs['input_width'])
        self.encode_method = kwargs['encode_method']
        # 1) Action embedding: each action ID → (unit_size)
        self.action_encoder = nn.Embedding(num_actions, unit_size)

        # 2) Optional projection from `unit_size` → `hidden_size`
        if use_linear:
            self.proj = nn.Linear(unit_size, hidden_size)

        # 3) State Embedding
        self.state_encoder = nn.Sequential(
            init_(nn.Conv2d(self.state_shape[0], 32, 3, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 4 * 4, hidden_size)), nn.ReLU())
        
        self.state_projector = nn.Linear(hidden_size, unit_size)

        # project 72 to 64
        self.POMDP_state_projector = nn.Linear(72, unit_size)



    def forward(self, s_h, a_h, s_h_len, a_h_len):
        """
        :param 
            s_h: shape (B, R, T, C, H, W), initial state of the environment.
            a_h: shape (B, R, T), each entry is an action ID.
        :return: shape (B, R, out_dim), where out_dim = hidden_size or unit_size
        """
        # Action sequence processing
        B, R, T = a_h.shape
        B_rolled = B * R
        a_h_flat = a_h.view(B_rolled, T).long()
        a_h_len_flat = a_h_len.view(B_rolled)

        embedded_actions = self.action_encoder(a_h_flat)  # [B*R, T, unit_size]

        # Extract initial state image
        s_0 = s_h[:, :, 0, :, :, :]  # (B, R, C, H, W)
        B, R, C, H, W = s_0.shape
        T += 1 #state has T+1 time steps
        BRT = B * R * T 

        s_0 = s_0.view(B_rolled, C, H, W)
        s0_embed = self.state_encoder(s_0)  # [B*R, hidden_size]
        s0_embed = self.state_projector(s0_embed)  # [B*R, unit_size]

        flat_sh = s_h.view(BRT, C, H, W)  # [B*R*T, C, H, W]
        s_embed = self.state_encoder(flat_sh)  # [B*R*T, hidden_size]
        s_embed = self.state_projector(s_embed)  # [B*R*T, unit_size]
        s_embed = s_embed.view(B_rolled, T, self.unit_size)  # [B*R, T, unit_size]

        

        if self.encode_method == 'fuse_s0':
            embedded_s_a = embedded_actions  # no s_0 prepended
            embedded_lengths = a_h_len_flat
        elif self.encode_method == 'prepend_s0':
            projected_s_0 = s0_embed.unsqueeze(1)  # [B*R, 1, unit_size]
            embedded_s_a = torch.cat((projected_s_0, embedded_actions), dim=1)
            embedded_lengths = a_h_len_flat + 1
        elif self.encode_method == 'sasa':
            # interleave s_embed and embedded_actions
            embedded_s_a = torch.zeros(B_rolled, 2*T-1, self.unit_size, device=s_embed.device)
            embedded_s_a[:, 0::2, :] = s_embed
            embedded_s_a[:, 1::2, :] = embedded_actions
            embedded_lengths = a_h_len_flat * 2 + 1



        # Pack and RNN
        packed_s_a = nn.utils.rnn.pack_padded_sequence(
            embedded_s_a, embedded_lengths.to("cpu"), batch_first=True, enforce_sorted=False
        )

        if self.rnn_type.upper() == 'GRU':
            packed_outputs, rnn_hxs = self.gru(packed_s_a)
        else:
            packed_outputs, (h_n, c_n) = self.lstm(packed_s_a)
            rnn_hxs = h_n

        final_hidden = rnn_hxs[-1]  # [B*R, unit_size]
        # Fuse with s_0 after RNN if enabled
        if self.encode_method == 'fuse_s0':
            # You can either add or concat — here we concat and project
            fused = final_hidden + s0_embed  # [B*R, unit_size]
            final_hidden = fused

        # Optional projection
        if self._use_linear:
            final_hidden = self.proj(final_hidden)  # [B*R, hidden_size]
            out_dim = self.hidden_size
        else:
            out_dim = self.unit_size



        final_hidden = final_hidden.view(B, R, out_dim)  # [B, R, out_dim]
        behavior_embedding = final_hidden.mean(dim=1)  # [B, out_dim]

        return behavior_embedding


class ActionBehaviorEncoderTransformer(ActionBehaviorEncoder):
    def __init__(
        self,
        recurrent,
        num_actions,
        hidden_size=64,
        rnn_type='GRU',
        dropout=0.0,
        use_linear=False,
        unit_size=256,
        **kwargs
    ):
        """
        Transformer-based encoder for action sequences.
        
        This class extends ActionBehaviorEncoder but replaces the RNN with a Transformer.
        """
        # Initialize the parent class without recurrent=True
        super(ActionBehaviorEncoderTransformer, self).__init__(
            recurrent=False,  # We'll use transformer instead
            num_actions=num_actions,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            dropout=dropout,
            use_linear=use_linear,
            unit_size=unit_size,
            **kwargs
        )
        
        # Get transformer parameters from kwargs with defaults
        self.fuse_s_0 = (kwargs['encode_method'] == 'fuse_s0')
        self.num_layers = kwargs['net']['transformer_layers']
        self.num_heads = kwargs['net']['transformer_heads']
        self.causal_attn = kwargs.get('causal_attention', True)
        self.max_demo_length = kwargs['max_demo_length']

        # positional embedding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 20*self.max_demo_length, unit_size))
        nn.init.normal_(self.pos_encoder, std=0.02)
        
        # end-of-sentence
        self.eos_token = nn.Parameter(torch.zeros(1, 1, unit_size))
        nn.init.normal_(self.eos_token, std=0.02)

        # separator token between rollouts
        self.sep_token = nn.Parameter(torch.zeros(1, 1, unit_size))
        nn.init.normal_(self.sep_token, std=0.02)
        
        # Layer norm before transformer
        self.pre_transformer_ln = nn.LayerNorm(unit_size)
        
        # Create transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=unit_size,
            nhead=self.num_heads,
            dim_feedforward=unit_size*4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )
        
        self.final_ln = nn.LayerNorm(unit_size)
        
        if use_linear:
            self.proj = nn.Linear(unit_size, hidden_size)

        self.POMDP = kwargs['POMDP']

    def forward(self, s_h, a_h, s_h_len, a_h_len, return_seq=False):
        """
        Encode a sequence of actions using a causal transformer with EOS token.
        
        :param s_h: shape (B, R, T, C, H, W) — B = programs, R = rollouts, T = time
        :param a_h: shape (B, R, T), each entry is an action ID
        :param s_h_len: sequence lengths for each rollout, shape (B, R)
        :param a_h_len: action sequence lengths, shape (B, R)
        :return: shape (B, out_dim), aggregated behavior embedding
        """
        # Initial state processing (similar to parent class)
        s_0 = s_h[:, :, 0, :, :, :]  # (B, R, T, C, H, W)
        B, R, C, H, W = s_0.shape
        B_rolled = B * R
        device = s_h.device

        new_s0 = s_0.view(B_rolled, C, H, W)
        if not self.POMDP:
            state_embeddings = self.state_encoder(new_s0)  # [B*R, hidden_size]
            state_embed = self.state_projector(state_embeddings)  # [B*R, unit_size]
        # Action sequence processing
        B, R, T = a_h.shape
        a_h_flat = a_h.view(B_rolled, T).long()
        a_h_len_flat = a_h_len.view(B_rolled)


        embedded_actions = self.action_encoder(a_h_flat)  # [B*R, T, unit_size]

        T += 1 
        BRT = B * R * T
        flat_sh = s_h.view(BRT, C, H, W)  # [B*R*T, C, H, W]
        if not self.POMDP:
            s_embed = self.state_encoder(flat_sh)  # [B*R*T, hidden_size]
            s_embed = self.state_projector(s_embed)  # [B*R*T, unit_size]
        else:
            flat_sh = flat_sh.view(BRT, C*H*W)  # [B*R*T, 72]
            s_embed = self.POMDP_state_projector(flat_sh)
        s_embed = s_embed.view(B_rolled, T, self.unit_size)  # [B*R, T, unit_size]
        
        # Handle state fusion
        if self.encode_method == 'fuse_s0':
            # Add +1 to make room for EOS token in all cases
            seq_len_T = T + 1
            # Create input with extra position for EOS
            transformer_input = torch.zeros(B_rolled, seq_len_T, self.unit_size, device=device)
            transformer_input[:, :T, :] = embedded_actions  # Place actions in first T positions
            seq_lengths = a_h_len_flat
        elif self.encode_method == 'concat_sasa':
            B, R, T = a_h.shape
            B_rolled = B * R
            device = s_embed.device
            unit_size = self.unit_size
            T += 1

            # Interleave S A S A ... to get (B*R, 2T-1, unit_size)
            embedded_s_a = torch.zeros(B_rolled, 2*T - 1, unit_size, device=device)
            embedded_s_a[:, 0::2, :] = s_embed
            embedded_s_a[:, 1::2, :] = embedded_actions

            # Actual lengths per rollout
            rollout_lengths = a_h_len_flat * 2 + 1  # shape: (B*R,)
            rollout_chunks = []
            batch_sep_token = self.sep_token.squeeze(0).squeeze(0)  # [unit_size]
            batch_eos_token = self.eos_token.squeeze(0).squeeze(0)  # [unit_size]

            max_seq_len = 0
            batch_sequences = []

            for b in range(B):
                # collect R rollouts for batch b
                rollout_seq = []
                total_len = 0
                for r in range(R):
                    idx = b * R + r
                    r_len = rollout_lengths[idx].item()
                    sas = embedded_s_a[idx, :r_len, :]  # (r_len, unit_size)
                    rollout_seq.append(sas)
                    total_len += r_len
                    if r != R - 1:
                        # Add sep_token between rollouts
                        rollout_seq.append(batch_sep_token.unsqueeze(0))  # (1, unit_size)
                        total_len += 1
                # After all rollouts, add EOS token
                rollout_seq.append(batch_eos_token.unsqueeze(0))  # (1, unit_size)
                total_len += 1

                full_seq = torch.cat(rollout_seq, dim=0)  # (total_len, unit_size)
                max_seq_len = max(max_seq_len, total_len)
                batch_sequences.append(full_seq)

            # Pad to max_seq_len
            transformer_input = torch.zeros(B, max_seq_len, unit_size, device=device)
            padding_mask = torch.ones(B, max_seq_len, dtype=torch.bool, device=device)  # default all padding

            for b, seq in enumerate(batch_sequences):
                L = seq.shape[0]
                transformer_input[b, :L, :] = seq
                padding_mask[b, :L] = False  # not padding

            # Apply layer norm
            seq_len = transformer_input.size(1)
            transformer_input = transformer_input + self.pos_encoder[:, :seq_len, :]
            transformer_input = self.pre_transformer_ln(transformer_input)

            # Generate causal mask if needed
            causal_mask = None
            if self.causal_attn:
                causal_mask = torch.triu(torch.full((max_seq_len, max_seq_len), float('-inf'), device=device), diagonal=1)

            # Use transformer with checkpointing
            transformer_output = checkpoint(
                self.transformer_encoder,
                transformer_input,
                causal_mask,
                padding_mask,
                self.causal_attn
            )

            # Layer norm again
            transformer_output = self.final_ln(transformer_output)

            # Grab embedding at EOS position (last non-padding token in each seq)
            eos_embeddings = []
            for b in range(B):
                eos_pos = (~padding_mask[b]).nonzero(as_tuple=False).max().item()
                eos_embeddings.append(transformer_output[b, eos_pos])

            final_hidden = torch.stack(eos_embeddings, dim=0)  # (B, unit_size)


            if self._use_linear:
                final_hidden = self.proj(final_hidden)

            if return_seq:
                return final_hidden, transformer_output, padding_mask  
            return final_hidden  # shape (B, hidden_size or unit_size)

        else:
             # Prepend s_0 to action sequence, add +1 for EOS
            seq_len_T = T + 2  # +1 for s_0, +1 for EOS
            transformer_input = torch.zeros(B_rolled, seq_len_T, self.unit_size, device=device)
            transformer_input[:, 0, :] = state_embed  # First position is state
            transformer_input[:, 1:T+1, :] = embedded_actions  # Then actions
            seq_lengths = a_h_len_flat + 1  # +1 for prepended state
        
        # Add EOS token at the end of each valid sequence
        for i in range(B_rolled):
            seq_len = seq_lengths[i].item()
            # Always place EOS token right after the valid sequence
            transformer_input[i, seq_len] = self.eos_token.squeeze(0).squeeze(0)

        # Create padding mask (values to be masked = True)
        # Everything after the EOS token should be masked
        padding_mask = torch.arange(seq_len_T, device=device).expand(B_rolled, seq_len_T) > (seq_lengths + 1).unsqueeze(1)
                
        # Apply layer normalization before transformer
        transformer_input = self.pre_transformer_ln(transformer_input)

        causal_mask = None
        if self.causal_attn:
            causal_mask = torch.triu(torch.ones(seq_len_T, seq_len_T, device=device) * float('-inf'), diagonal=1)
        
        # Apply transformer encoder with native causal masking
        # transformer_output = self.transformer_encoder(
        #     src=transformer_input,
        #     mask=causal_mask,
        #     src_key_padding_mask=padding_mask,
        #     is_causal=self.causal_attn  # Native causal masking is sufficient
        # )  # [B*R, T, unit_size]
        
        transformer_output = checkpoint(
            self.transformer_encoder,
            transformer_input,
            causal_mask,
            padding_mask,
            self.causal_attn
        )
        
        # Apply final layer norm
        transformer_output = self.final_ln(transformer_output)
        
        # Extract embeddings at the end of each valid sequence (including EOS)
        sequence_embeddings = []
        for i in range(B_rolled):
            # Get the actual sequence length (might be shorter than T)
            seq_len = seq_lengths[i].item()
            # Use the EOS token position (or last valid position if no EOS)
            seq_len = min(seq_len, T-1)  # Ensure index is valid
            sequence_embeddings.append(transformer_output[i, seq_len])
        
        # Stack to get final embeddings
        final_hidden = torch.stack(sequence_embeddings)  # [B*R, unit_size]
        
        # Fuse with s_0 after transformer if enabled
        if self.fuse_s_0:
            # Similar to the parent class, we add state_embed to final_hidden
            fused = final_hidden + state_embed  # [B*R, unit_size]
            final_hidden = fused
        
        # Optional projection: [B*R, hidden_size]
        if self._use_linear:
            final_hidden = self.proj(final_hidden)
            out_dim = self.hidden_size
        else:
            out_dim = self.unit_size

        # Reshape to [B, R, out_dim]
        final_hidden = final_hidden.reshape(B, R, out_dim)
        
        # Mean over R rollouts: [B, out_dim]
        behavior_embedding = final_hidden.mean(dim=1)
        
        return behavior_embedding




class StateBehaviorEncoder(NNBase):
    def __init__(
        self,
        recurrent,
        num_actions,
        hidden_size=64,
        rnn_type='GRU',
        dropout=0.0,
        use_linear=False,
        unit_size=256,
        **kwargs
    ):
        """
        Encodes a rollout of actions (only) into a single latent vector per rollout.

        :param num_actions: how many possible actions exist (for embedding).
        :param hidden_size: final output dimension (matching ProgramEncoder).
        :param rnn_type: 'GRU' or 'LSTM'.
        :param dropout: dropout on the RNN.
        :param use_linear: if True, project from `unit_size` down to `hidden_size`.
        :param unit_size: internal dimension for the RNN and/or embedding.
        """
        # We set recurrent=True so NNBase will handle RNN init for us
        super(StateBehaviorEncoder, self).__init__(
            recurrent=True,
            recurrent_input_size=unit_size,   # dimension fed into the RNN
            hidden_size=unit_size,  # RNN’s hidden dimension
            dropout=dropout,
            rnn_type=rnn_type
        )

        # Same style init as ConditionPolicy, etc.
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu')
        )

        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self._use_linear = use_linear
        self.unit_size = unit_size
        self.state_shape = (kwargs['input_channel'], kwargs['input_height'], kwargs['input_width'])
        # 1) Action embedding: each action ID → (unit_size)
        self.action_encoder = nn.Embedding(num_actions, unit_size)

        # 2) Optional projection from `unit_size` → `hidden_size`
        if use_linear:
            self.proj = nn.Linear(unit_size, hidden_size)

        # 3) State Embedding
        self.state_encoder = nn.Sequential(
            init_(nn.Conv2d(self.state_shape[0], 32, 3, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 4 * 4, hidden_size)), nn.ReLU())
        
        self.state_projector = nn.Linear(hidden_size, unit_size)

    def forward(self, s_h, a_h, s_h_len, a_h_len):
        """
        Encode a sequence of states using an RNN.
        
        :param s_h: shape (B, R, T, C, H, W) — B = programs, R = rollouts, T = time
        :param a_h: not used, just pass for compatibility
        :return: shape (B, out_dim), aggregated behavior embedding
        """
        B, R, T, C, H, W = s_h.shape
        BR = B * R

        # Flatten to [BR*T, C, H, W] to pass through ConvNet
        s_h = s_h.view(BR * T, C, H, W)  # [BR*T, C, H, W]
        state_embeddings = self.state_encoder(s_h)  # [BR*T, hidden_size]

        # Project to RNN input dim: [BR*T, unit_size]
        projected_state = self.state_projector(state_embeddings)

        # Reshape to sequence form: [BR, T, unit_size]
        projected_state_seq = projected_state.view(BR, T, self.unit_size)
        s_h_len_flat = s_h_len.view(BR)

        # Transpose for RNN: [T, BR, unit_size]
        # embedded_s = projected_state_seq.transpose(0, 1)
        packed_s = pack_padded_sequence(
            projected_state_seq, 
            lengths=s_h_len_flat.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )

        # Run RNN
        if self.rnn_type.upper() == 'GRU':
            packed_outputs, rnn_hxs = self.gru(packed_s)  # rnn_hxs: [num_layers, BR, unit_size]
        else:
            packed_outputs, (h_n, c_n) = self.lstm(packed_s)
            rnn_hxs = h_n

        # Get last hidden state from top layer: [BR, unit_size]
        final_hidden = rnn_hxs[-1]

        # Optional projection: [BR, hidden_size]
        if self._use_linear:
            final_hidden = self.proj(final_hidden)
            out_dim = self.hidden_size
        else:
            out_dim = self.unit_size

        # Reshape back to [B, R, out_dim]
        final_hidden = final_hidden.view(B, R, out_dim)

        # Mean over R rollouts: [B, out_dim]
        behavior_embedding = final_hidden.mean(dim=1)
        return behavior_embedding



class ProgramEncoder(NNBase):
    def __init__(self, num_inputs, num_outputs, recurrent=True, hidden_size=64, rnn_type='GRU', two_head=False, dropout=0.0, use_linear=False, unit_size=256):
        super(ProgramEncoder, self).__init__(recurrent, num_inputs, unit_size, dropout, rnn_type)

        self._rnn_type = rnn_type
        self._two_head = two_head
        self.token_encoder = nn.Embedding(num_inputs, num_inputs)
        self._use_linear = use_linear
        
        # Add Linear Layer to compress latent
        if self._use_linear:
            self.fc = nn.Linear(unit_size, hidden_size)

    def forward(self, src, src_len):
        program_embeddings = self.token_encoder(src)
        src_len = src_len.cpu()
        packed_embedded = pack_padded_sequence(program_embeddings, src_len, batch_first=True,
                                                            enforce_sorted=False)

        if self.is_recurrent:
            x, rnn_hxs = self.gru(packed_embedded)
        
        # Add Linear Layer to compress latent
        if self._use_linear:
            rnn_hxs = self.fc(rnn_hxs)

        return x, rnn_hxs

class Decoder(NNBase):
    def __init__(self, num_inputs, num_outputs, recurrent=False, hidden_size=64, rnn_type='GRU', two_head=False, dropout=0.0, unit_size=256, **kwargs):
        super(Decoder, self).__init__(recurrent, num_inputs+unit_size, unit_size, dropout, rnn_type)

        self._rnn_type = rnn_type
        self._two_head = two_head
        self.num_inputs = num_inputs
        self.use_simplified_dsl = kwargs['dsl']['use_simplified_dsl']
        self.max_program_len = kwargs['dsl']['max_program_len']
        self.grammar = kwargs['grammar']
        self.num_program_tokens = kwargs['num_program_tokens']
        self.setup = kwargs['algorithm']
        self.setup = 'CEM' if self.setup == 'CEM_transfer' else self.setup
        self.rl_algorithm = kwargs['rl']['algo']['name']
        self.value_method = kwargs['rl']['value_method']
        self.value_embedding = 'eop_rnn'
        print("Decoder max_program_len: ", self.max_program_len)
        print("Decoder dropout: ", dropout)
        print("Decoder use_simplified_dsl: ", self.use_simplified_dsl)
        print("Decoder num_outputs (token space): ", num_outputs)
        print("Decoder num_program_tokens: ", self.num_program_tokens)
    
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.token_encoder = nn.Embedding(num_inputs, num_inputs)

        self.token_output_layer = nn.Sequential(
            init_(nn.Linear(unit_size + num_inputs + unit_size, unit_size)), nn.Tanh(),
            init_(nn.Linear(unit_size, num_outputs)))

        # This check is required only to support backward compatibility to pre-trained models
        if (self.setup =='RL' or self.setup =='supervisedRL') and kwargs['rl']['algo']['name'] != 'reinforce':
            self.critic = nn.Sequential(
                init_(nn.Linear(unit_size, unit_size)), nn.Tanh(),
                init_(nn.Linear(unit_size, unit_size)), nn.Tanh())

            self.critic_linear = init_(nn.Linear(unit_size, 1))

        if self._two_head:
            self.eop_output_layer = nn.Sequential(
                init_(nn.Linear(unit_size + num_inputs + unit_size, unit_size)), nn.Tanh(),
                init_(nn.Linear(unit_size, 2)))

        self._init_syntax_checker(kwargs)

        self.softmax = nn.LogSoftmax(dim=-1)

        # Add Linear Layer to uncompress latent
        self._use_linear = kwargs['net']['use_linear']
        if self._use_linear:
            self.fc = nn.Linear(hidden_size, kwargs['net']['num_rnn_decoder_units'])

        self.train()

    def _init_syntax_checker(self, config):
        # use syntax checker to check grammar of output program prefix
        if self.use_simplified_dsl:
            self.prl_tokens = config['prl_tokens']
            self.dsl_tokens = config['dsl_tokens']
            self.prl2dsl_mapping = config['prl2dsl_mapping']
            syntax_checker_tokens = copy.copy(config['prl_tokens'])
        else:
            syntax_checker_tokens = copy.copy(config['dsl_tokens'])
        
        T2I = {token: i for i, token in enumerate(syntax_checker_tokens)}
        T2I['<pad>'] = len(syntax_checker_tokens)
        self.T2I = T2I
        syntax_checker_tokens.append('<pad>')
        
        print("Decoder _init_syntax_checker len: ", len(syntax_checker_tokens))
        
        if self.grammar == 'handwritten':
            self.syntax_checker = PySyntaxChecker(T2I, use_cuda='cuda' in config['device'],
                                                  use_simplified_dsl=self.use_simplified_dsl,
                                                  new_tokens=syntax_checker_tokens)

    def _forward_one_pass(self, current_tokens, context, rnn_hxs, masks):
        token_embedding = self.token_encoder(current_tokens)
        inputs = torch.cat((token_embedding, context), dim=-1)

        if self.is_recurrent:
            outputs, rnn_hxs = self._forward_rnn(inputs, rnn_hxs, masks.view(-1, 1))

        pre_output = torch.cat([outputs, token_embedding, context], dim=1)
        output_logits = self.token_output_layer(pre_output)

        value = None
        if (self.setup =='RL' or self.setup =='supervisedRL') and self.rl_algorithm != 'reinforce':
            hidden_critic = self.critic(rnn_hxs)
            value = self.critic_linear(hidden_critic)

        eop_output_logits = None
        if self._two_head:
            eop_output_logits = self.eop_output_layer(pre_output)
        return value, output_logits, rnn_hxs, eop_output_logits

    def _temp_init(self, batch_size, device):
        # create input with token as DEF
        inputs = torch.ones((batch_size)).to(torch.long).to(device)
        inputs = (0 * inputs)# if self.use_simplified_dsl else (2 * inputs)

        # input to the GRU
        gru_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        return inputs, gru_mask

    def _get_syntax_mask(self, batch_size, current_tokens, mask_size, grammar_state):
        out_of_syntax_list = []
        device = current_tokens.device
        out_of_syntax_mask = torch.zeros((batch_size, mask_size),
                                         dtype=torch.bool, device=device)

        for program_idx, inp_token in enumerate(current_tokens):
            inp_dsl_token = inp_token.detach().cpu().numpy().item()
            out_of_syntax_list.append(self.syntax_checker.get_sequence_mask(grammar_state[program_idx],[inp_dsl_token]).to(device))

        torch.cat(out_of_syntax_list, 0, out=out_of_syntax_mask)
        out_of_syntax_mask = out_of_syntax_mask.squeeze()
        syntax_mask = torch.where(out_of_syntax_mask,
                                  -torch.finfo(torch.float32).max * torch.ones_like(out_of_syntax_mask).float(),
                                  torch.zeros_like(out_of_syntax_mask).float())

        # If m) is not part of next valid tokens in syntax_mask then only eop action can be eop=0 otherwise not
        # use absence of m) to mask out eop = 1, use presence of m) and eop=1 to mask out all tokens except m)
        eop_syntax_mask = None
        if self._two_head:
            # use absence of m) to mask out eop = 1
            gather_m_closed = torch.tensor(batch_size * [self.T2I['m)']], dtype=torch.long, device=device).view(-1, 1)
            eop_in_valid_set = torch.gather(syntax_mask, 1, gather_m_closed)
            eop_syntax_mask = torch.zeros((batch_size, 2), device=device)
            # if m) is absent we can't predict eop=1
            eop_syntax_mask[:, 1] = eop_in_valid_set.flatten()

        return syntax_mask, eop_syntax_mask, grammar_state

    def _get_eop_preds(self, eop_output_logits, eop_syntax_mask, syntax_mask, output_mask, deterministic=False):
        batch_size = eop_output_logits.shape[0]
        device = eop_output_logits.device

        # eop_action
        if eop_syntax_mask is not None:
            assert eop_output_logits.shape == eop_syntax_mask.shape
            eop_output_logits += eop_syntax_mask
        if self.setup == 'supervised':
            eop_preds = self.softmax(eop_output_logits).argmax(dim=-1).to(torch.bool)
        elif self.setup == 'RL':
            # define distribution over current logits
            eop_dist = FixedCategorical(logits=eop_output_logits)
            # sample actions
            eop_preds = eop_dist.mode() if deterministic else eop_dist.sample()
        else:
            raise NotImplementedError()


        #  use presence of m) and eop=1 to mask out all tokens except m)
        if self.grammar != 'None':
            new_output_mask = (~(eop_preds.to(torch.bool))) * output_mask
            assert output_mask.dtype == torch.bool
            output_mask_change = (new_output_mask != output_mask).view(-1, 1)
            output_mask_change_repeat = output_mask_change.repeat(1, syntax_mask.shape[1])
            new_syntax_mask = -torch.finfo(torch.float32).max * torch.ones_like(syntax_mask).float()
            new_syntax_mask[:, self.T2I['m)']] = 0
            syntax_mask = torch.where(output_mask_change_repeat, new_syntax_mask, syntax_mask)

        return eop_preds, eop_output_logits, syntax_mask

    def forward(self, gt_programs, embeddings, teacher_enforcing=True, action=None, output_mask_all=None,
                eop_action=None, deterministic=False, evaluate=False, max_program_len=float('inf')):
        if self.setup == 'supervised':
            assert deterministic == True
        batch_size, device = embeddings.shape[0], embeddings.device
        # NOTE: for pythorch >=1.2.0, ~ only works correctly on torch.bool
        if evaluate:
            output_mask = output_mask_all[:, 0]
        else:
            output_mask = torch.ones(batch_size).to(torch.bool).to(device)

        current_tokens, gru_mask = self._temp_init(batch_size, device)

        # Add Linear Layer to uncompress latent
        if self._use_linear:
            embeddings = self.fc(embeddings)

        if self._rnn_type == 'GRU':
            rnn_hxs = embeddings
        elif self._rnn_type == 'LSTM':
            rnn_hxs = (embeddings, embeddings)
        else:
            raise NotImplementedError()

        # Encode programs
        max_program_len = min(max_program_len, self.max_program_len)
        value_all = []
        pred_programs = []
        pred_programs_log_probs_all = []
        dist_entropy_all = []
        eop_dist_entropy_all = []
        output_logits_all = []
        eop_output_logits_all = []
        eop_pred_programs = []
        if not evaluate:
            output_mask_all = torch.ones(batch_size, self.max_program_len).to(torch.bool).to(device)
        first_end_token_idx = self.max_program_len * torch.ones(batch_size).to(device)

        # using get_initial_checker_state2 because we skip prediction for 'DEF', 'run' tokens
        if self.grammar == 'handwritten':
            if self.use_simplified_dsl:
                grammar_state = [self.syntax_checker.get_initial_checker_state2()
                                 for _ in range(batch_size)]
            else:
                grammar_state = [self.syntax_checker.get_initial_checker_state()
                                 for _ in range(batch_size)]

        for i in range(max_program_len):
            value, output_logits, rnn_hxs, eop_output_logits = self._forward_one_pass(current_tokens, embeddings,
                                                                                      rnn_hxs, gru_mask)

            # limit possible actions using syntax checker if available
            # action_logits * syntax_mask where syntax_mask = {-inf, 0}^|num_program_tokens|
            # syntax_mask = 0  for action a iff for given input(e.g.'DEF'), a(e.g.'run') creates a valid program prefix
            syntax_mask = None
            eop_syntax_mask = None
            if self.grammar != 'None':
                mask_size = output_logits.shape[1]
                syntax_mask, eop_syntax_mask, grammar_state = self._get_syntax_mask(batch_size, current_tokens,
                                                                                    mask_size, grammar_state)

            # get eop action and new syntax mask if using syntax checker
            if self._two_head:
                eop_preds, eop_output_logits, syntax_mask = self._get_eop_preds(eop_output_logits, eop_syntax_mask,
                                                                                syntax_mask, output_mask_all[:, i])

            # apply softmax
            if syntax_mask is not None:
                assert (output_logits.shape == syntax_mask.shape), '{}:{}'.format(output_logits.shape, syntax_mask.shape)
                output_logits += syntax_mask
            if self.setup == 'supervised' or self.setup == 'CEM' or self.setup == 'PPO_option' or self.setup == 'SAC_option':
                preds = self.softmax(output_logits).argmax(dim=-1)
            elif self.setup == 'RL':
                # define distribution over current logits
                dist = FixedCategorical(logits=output_logits)
                # sample actions
                preds = dist.mode().squeeze() if deterministic else dist.sample().squeeze()
                # calculate log probabilities
                if evaluate:
                    assert action[:,i].shape == preds.shape
                    pred_programs_log_probs = dist.log_probs(action[:,i])
                else:
                    pred_programs_log_probs = dist.log_probs(preds)

                if self._two_head:
                    raise NotImplementedError()
                # calculate entropy
                dist_entropy = dist.entropy()
                if self._two_head:
                    raise NotImplementedError()
                pred_programs_log_probs_all.append(pred_programs_log_probs)
                dist_entropy_all.append(dist_entropy.view(-1, 1))
            else:
                raise NotImplementedError()

            # calculate mask for current tokens
            assert preds.shape == output_mask.shape
            if not evaluate:
                if self._two_head:
                    output_mask = (~(eop_preds.to(torch.bool))) * output_mask
                else:
                    output_mask = (~((preds == self.num_program_tokens - 1).to(torch.bool))) * output_mask

                # recalculate first occurrence of <pad> for each program
                first_end_token_idx = torch.min(first_end_token_idx,
                                                ((self.max_program_len * output_mask.float()) +
                                                 ((1 - output_mask.float()) * i)).flatten())

            value_all.append(value)
            output_logits_all.append(output_logits)
            pred_programs.append(preds)
            if self._two_head:
                eop_output_logits_all.append(eop_output_logits)
                eop_pred_programs.append(eop_preds)
            if not evaluate:
                output_mask_all[:, i] = output_mask.flatten()

            if self.setup == 'supervised':
                if teacher_enforcing:
                    current_tokens = gt_programs[:, i+1].squeeze()
                else:
                    current_tokens = preds.squeeze()
            else:
                if evaluate:
                    assert self.setup == 'RL'
                    current_tokens = action[:, i]
                else:
                    current_tokens = preds.squeeze()


        # umask first end-token for two headed policy
        if not evaluate:
            output_mask_all = _unmask_idx(output_mask_all, first_end_token_idx, self.max_program_len).detach()

        # combine all token parameters to get program parameters
        raw_pred_programs_all = torch.stack(pred_programs, dim=1)
        raw_output_logits_all = torch.stack(output_logits_all, dim=1)
        pred_programs_len = torch.sum(output_mask_all, dim=1, keepdim=True)

        if not self._two_head:
            assert output_mask_all.dtype == torch.bool
            pred_programs_all = torch.where(output_mask_all, raw_pred_programs_all,
                                            int(self.num_program_tokens - 1) * torch.ones_like(raw_pred_programs_all))
            eop_pred_programs_all = -1 * torch.ones_like(pred_programs_all)
            raw_eop_output_logits_all = None
        else:
            pred_programs_all = raw_pred_programs_all
            eop_pred_programs_all = torch.stack(eop_pred_programs, dim=1)
            raw_eop_output_logits_all = torch.stack(eop_output_logits_all, dim=1)

        # calculate log_probs, value, actions for program from token values
        if self.setup == 'RL':
            raw_pred_programs_log_probs_all = torch.cat(pred_programs_log_probs_all, dim=1)
            pred_programs_log_probs_all = masked_sum(raw_pred_programs_log_probs_all, output_mask_all,
                                                     dim=1, keepdim=True)

            raw_dist_entropy_all = torch.cat(dist_entropy_all, dim=1)
            dist_entropy_all = masked_mean(raw_dist_entropy_all, output_mask_all,
                                           dim=tuple(range(len(output_mask_all.shape))))

            # calculate value for program from token values
            if self.rl_algorithm != 'reinforce':
                if self.value_method == 'mean':
                    raw_value_all = torch.cat(value_all, dim=1)
                    value_all = masked_mean(raw_value_all, output_mask_all, dim=1, keepdim=True)
                else:
                    # calculate value function from hidden states
                    raw_value_all = torch.cat(value_all, dim=1)
                    value_idx = torch.sum(output_mask_all, dim=1, keepdim=True) - 1
                    assert len(value_idx.shape) == 2 and value_idx.shape[1] == 1
                    value_all = torch.gather(raw_value_all, 1, value_idx)

                    # This value calculation is just for sanity check
                    with torch.no_grad():
                        value_idx_2 = first_end_token_idx.clamp(max=self.max_program_len - 1).long().reshape(-1, 1)
                        value_all_2 = torch.gather(raw_value_all, 1, value_idx_2)
                        assert torch.sum(value_all != value_all_2) == 0
                    assert value_all.shape[0] == batch_size
            else:
                value_all = torch.zeros_like(pred_programs_log_probs_all)
        else:
            dist_entropy_all = None
            value_all = None

        return value_all, pred_programs_all, pred_programs_len, pred_programs_log_probs_all, raw_output_logits_all,\
               eop_pred_programs_all, raw_eop_output_logits_all, output_mask_all, dist_entropy_all
               
               
class DecoderTransformer(NNBase):
    def __init__(self, num_inputs, num_outputs, recurrent=False, hidden_size=64, rnn_type='GRU', two_head=False, dropout=0.0, unit_size=256, **kwargs):
        # Use False for recurrent since we're using a transformer
        super(DecoderTransformer, self).__init__(False, num_inputs+unit_size, unit_size, dropout, rnn_type)

        self._rnn_type = rnn_type
        self._two_head = two_head
        self.num_inputs = num_inputs
        self.use_simplified_dsl = kwargs['dsl']['use_simplified_dsl']
        self.max_program_len = kwargs['dsl']['max_program_len']
        self.grammar = kwargs['grammar']
        self.num_program_tokens = kwargs['num_program_tokens']
        self.setup = kwargs['algorithm']
        self.setup = 'CEM' if self.setup == 'CEM_transfer' else self.setup
        self.rl_algorithm = kwargs['rl']['algo']['name']
        self.value_method = kwargs['rl']['value_method']
        self.value_embedding = 'eop_rnn'

        print("DecoderTransformer max_program_len: ", self.max_program_len)
        print("DecoderTransformer dropout: ", dropout)
        print("DecoderTransformer use_simplified_dsl: ", self.use_simplified_dsl)
        print("DecoderTransformer num_outputs (token space): ", num_outputs)
        print("DecoderTransformer num_program_tokens: ", self.num_program_tokens)
    
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        # Token embedding layer
        self.token_encoder = nn.Embedding(num_inputs, unit_size)
        
        # Positional embedding
        self.pos_encoder = nn.Parameter(torch.zeros(1, self.max_program_len, unit_size))
        nn.init.normal_(self.pos_encoder, std=0.02)
        
        # Transformer layers configuration
        self.num_layers = kwargs.get('net', {}).get('transformer_decoder_layers', 6)
        self.num_heads = kwargs.get('net', {}).get('transformer_decoder_heads', 8)
        self.use_behavior_memory = kwargs['net'].get('use_behavior_memory', True)
        
        # Create TransformerDecoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=unit_size,
            nhead=self.num_heads,
            dim_feedforward=unit_size*4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-norm architecture for better stability
        )
        
        # TransformerDecoder (processes tokens sequentially with attention)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=self.num_layers
        )
        
        # Output projection layer (from unit_size to vocabulary size)
        self.token_output_layer = nn.Linear(unit_size, num_outputs)
        
        # Layer norms
        self.pre_decoder_ln = nn.LayerNorm(unit_size)
        self.final_ln = nn.LayerNorm(unit_size)

        # For critic network in RL setup
        if (self.setup =='RL' or self.setup =='supervisedRL') and kwargs['rl']['algo']['name'] != 'reinforce':
            self.critic = nn.Sequential(
                init_(nn.Linear(unit_size, unit_size)), nn.Tanh(),
                init_(nn.Linear(unit_size, unit_size)), nn.Tanh())

            self.critic_linear = init_(nn.Linear(unit_size, 1))

        # For two-head policy (separate end-of-program prediction)
        if self._two_head:
            self.eop_output_layer = nn.Sequential(
                init_(nn.Linear(unit_size, unit_size)), nn.Tanh(),
                init_(nn.Linear(unit_size, 2)))

        # Initialize syntax checker for grammar-based decoding
        self._init_syntax_checker(kwargs)

        self.softmax = nn.LogSoftmax(dim=-1)

        # Add Linear Layer to uncompress latent
        self._use_linear = kwargs['net']['use_linear']
        if self._use_linear:
            self.fc = nn.Linear(hidden_size, unit_size)

        self.train()

    def _init_syntax_checker(self, config):
        # use syntax checker to check grammar of output program prefix
        if self.use_simplified_dsl:
            self.prl_tokens = config['prl_tokens']
            self.dsl_tokens = config['dsl_tokens']
            self.prl2dsl_mapping = config['prl2dsl_mapping']
            syntax_checker_tokens = copy.copy(config['prl_tokens'])
        else:
            syntax_checker_tokens = copy.copy(config['dsl_tokens'])
        
        T2I = {token: i for i, token in enumerate(syntax_checker_tokens)}
        T2I['<pad>'] = len(syntax_checker_tokens)
        self.T2I = T2I
        syntax_checker_tokens.append('<pad>')
        
        print("DecoderTransformer _init_syntax_checker len: ", len(syntax_checker_tokens))
        
        if self.grammar == 'handwritten':
            self.syntax_checker = PySyntaxChecker(T2I, use_cuda='cuda' in config['device'],
                                                 use_simplified_dsl=self.use_simplified_dsl,
                                                 new_tokens=syntax_checker_tokens)

    def generate_square_subsequent_mask(self, sz, device):
        """
        Generate a square mask for the sequence to prevent attending to future positions.
        The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _forward_one_pass(self, current_tokens, context, transformer_outputs, step_idx, causal_mask, behavior_memory=None, behavior_padding_mask=None):
        """
        Process a single decoding step using the transformer decoder
        
        Args:
            current_tokens: Current token IDs for the batch
            context: The latent embedding vectors
            transformer_outputs: Previous transformer outputs to use as memory
            step_idx: Current decoding step index
            causal_mask: Causal mask for self-attention
            
        Returns:
            value: Value prediction (for RL)
            output_logits: Token logits
            transformer_outputs: Updated transformer memory
            eop_output_logits: End-of-program logits (if two_head=True)
        """
        if step_idx == 0:
            if self.use_behavior_memory and behavior_memory is not None:
                lat_token = context.unsqueeze(1)              # [B, 1, d]
                memory    = torch.cat([lat_token, behavior_memory], dim=1)
                mem_key_padding_mask = torch.cat(
                    [torch.zeros(B, 1, dtype=torch.bool, device=device),  # never mask LAT
                    behavior_padding_mask], dim=1)
            else:
                memory = context.unsqueeze(1)               # legacy path
                mem_key_padding_mask = None
            self._cached_memory = (memory, mem_key_padding_mask)

        memory, mem_key_padding_mask = self._cached_memory   # <-- keep!

        
        device = current_tokens.device
        
        # Embed current tokens and add positional encoding
        token_embedding = self.token_encoder(current_tokens)  # [batch_size, unit_size]
        
        # Create target sequence for the decoder (with positional info)
        if transformer_outputs is None:
            # First step: just use the current token embedding
            decoder_input = token_embedding.unsqueeze(1)  # [batch_size, 1, unit_size]
            pos_embeddings = self.pos_encoder[:, 0:1, :]  # [1, 1, unit_size]
            decoder_input = decoder_input + pos_embeddings
        else:
            # For later steps, use all previous outputs plus new token
            decoder_input = torch.cat([transformer_outputs, token_embedding.unsqueeze(1)], dim=1)
            pos_embeddings = self.pos_encoder[:, :decoder_input.size(1), :]
            decoder_input = decoder_input + pos_embeddings
            
        # Apply layer norm before decoder
        decoder_input = self.pre_decoder_ln(decoder_input)
        
        # Create attention mask that prevents attending to subsequent positions
        if causal_mask is None and decoder_input.size(1) > 1:
            causal_mask = self.generate_square_subsequent_mask(decoder_input.size(1), device)
            

        
        # Run transformer decoder
        # For first step, there's no self-attention mask
        # transformer_out = self.transformer_decoder(
        #     tgt=decoder_input,  # Target sequence (tokens so far)
        #     memory=memory,      # Memory from encoder (latent embedding)
        #     tgt_mask=causal_mask  # Causal mask for self-attention
        # )
        transformer_out = checkpoint(
            self.transformer_decoder,
            decoder_input,               # tgt
            memory,                      # memory
            causal_mask,                 # tgt_mask
            None,                        # memory_mask
            None,                        # tgt_key_padding_mask
            mem_key_padding_mask         # memory_key_padding_mask
        )
        
        # Apply final layer norm
        transformer_out = self.final_ln(transformer_out)
        
        # Get last token's output for prediction
        last_token_output = transformer_out[:, -1]  # [batch_size, unit_size]
        
        # Generate logits for next token prediction
        output_logits = self.token_output_layer(last_token_output)
        
        # For RL, calculate value prediction
        value = None
        if (self.setup =='RL' or self.setup =='supervisedRL') and self.rl_algorithm != 'reinforce':
            hidden_critic = self.critic(last_token_output)
            value = self.critic_linear(hidden_critic)
            
        # For two-head policy, predict end-of-program token separately
        eop_output_logits = None
        if self._two_head:
            eop_output_logits = self.eop_output_layer(last_token_output)
            
        return value, output_logits, transformer_out, eop_output_logits

    def _temp_init(self, batch_size, device):
        # create input with token as DEF
        inputs = torch.ones((batch_size)).to(torch.long).to(device)
        inputs = (0 * inputs)  # Start with DEF token (ID 0)

        # Create mask (all ones since we're starting)
        gru_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        return inputs, gru_mask

    def _get_syntax_mask(self, batch_size, current_tokens, mask_size, grammar_state):
        out_of_syntax_list = []
        device = current_tokens.device
        # Create output tensor with correct shape [batch_size, 1, mask_size]
        out_of_syntax_mask = torch.zeros((batch_size, 1, mask_size),
                                         dtype=torch.bool, device=device)

        for program_idx, inp_token in enumerate(current_tokens):
            inp_dsl_token = inp_token.detach().cpu().numpy().item()
            out_of_syntax_list.append(self.syntax_checker.get_sequence_mask(grammar_state[program_idx],[inp_dsl_token]).to(device))

        torch.cat(out_of_syntax_list, 0, out=out_of_syntax_mask)
        out_of_syntax_mask = out_of_syntax_mask.squeeze()
        syntax_mask = torch.where(out_of_syntax_mask,
                                  -torch.finfo(torch.float32).max * torch.ones_like(out_of_syntax_mask).float(),
                                  torch.zeros_like(out_of_syntax_mask).float())

        # If m) is not part of next valid tokens in syntax_mask then only eop action can be eop=0 otherwise not
        # use absence of m) to mask out eop = 1, use presence of m) and eop=1 to mask out all tokens except m)
        eop_syntax_mask = None
        if self._two_head:
            # use absence of m) to mask out eop = 1
            gather_m_closed = torch.tensor(batch_size * [self.T2I['m)']], dtype=torch.long, device=device).view(-1, 1)
            eop_in_valid_set = torch.gather(syntax_mask, 1, gather_m_closed)
            eop_syntax_mask = torch.zeros((batch_size, 2), device=device)
            # if m) is absent we can't predict eop=1
            eop_syntax_mask[:, 1] = eop_in_valid_set.flatten()

        return syntax_mask, eop_syntax_mask, grammar_state

    def _get_eop_preds(self, eop_output_logits, eop_syntax_mask, syntax_mask, output_mask, deterministic=False):
        batch_size = eop_output_logits.shape[0]
        device = eop_output_logits.device

        # eop_action
        if eop_syntax_mask is not None:
            assert eop_output_logits.shape == eop_syntax_mask.shape
            eop_output_logits += eop_syntax_mask
        if self.setup == 'supervised':
            eop_preds = self.softmax(eop_output_logits).argmax(dim=-1).to(torch.bool)
        elif self.setup == 'RL':
            # define distribution over current logits
            eop_dist = FixedCategorical(logits=eop_output_logits)
            # sample actions
            eop_preds = eop_dist.mode() if deterministic else eop_dist.sample()
        else:
            raise NotImplementedError()

        #  use presence of m) and eop=1 to mask out all tokens except m)
        if self.grammar != 'None':
            new_output_mask = (~(eop_preds.to(torch.bool))) * output_mask
            assert output_mask.dtype == torch.bool
            output_mask_change = (new_output_mask != output_mask).view(-1, 1)
            output_mask_change_repeat = output_mask_change.repeat(1, syntax_mask.shape[1])
            new_syntax_mask = -torch.finfo(torch.float32).max * torch.ones_like(syntax_mask).float()
            new_syntax_mask[:, self.T2I['m)']] = 0
            syntax_mask = torch.where(output_mask_change_repeat, new_syntax_mask, syntax_mask)

        return eop_preds, eop_output_logits, syntax_mask

    def forward(self, gt_programs, embeddings, teacher_enforcing=True, action=None, output_mask_all=None,
                eop_action=None, deterministic=False, evaluate=False, max_program_len=float('inf'), behavior_memory=None,
                behavior_padding_mask=None, ):
        if self.setup == 'supervised':
            assert deterministic == True
        batch_size, device = embeddings.shape[0], embeddings.device
        
        # NOTE: for pythorch >=1.2.0, ~ only works correctly on torch.bool
        if evaluate:
            output_mask = output_mask_all[:, 0]
        else:
            output_mask = torch.ones(batch_size).to(torch.bool).to(device)

        current_tokens, gru_mask = self._temp_init(batch_size, device)

        # Add Linear Layer to uncompress latent if needed
        if self._use_linear:
            embeddings = self.fc(embeddings)

        # Encode programs
        max_program_len = min(max_program_len, self.max_program_len)
        value_all = []
        pred_programs = []
        pred_programs_log_probs_all = []
        dist_entropy_all = []
        eop_dist_entropy_all = []
        output_logits_all = []
        eop_output_logits_all = []
        eop_pred_programs = []
        
        if not evaluate:
            output_mask_all = torch.ones(batch_size, self.max_program_len).to(torch.bool).to(device)
        first_end_token_idx = self.max_program_len * torch.ones(batch_size).to(device)

        # using get_initial_checker_state2 because we skip prediction for 'DEF', 'run' tokens
        if self.grammar == 'handwritten':
            if self.use_simplified_dsl:
                grammar_state = [self.syntax_checker.get_initial_checker_state2()
                                for _ in range(batch_size)]
            else:
                grammar_state = [self.syntax_checker.get_initial_checker_state()
                                for _ in range(batch_size)]

        # Initialize transformer outputs to None
        transformer_outputs = None
        causal_mask = None

        for i in range(max_program_len):
            # Run one step of the transformer decoder
            value, output_logits, transformer_outputs, eop_output_logits = self._forward_one_pass(
                current_tokens, embeddings, transformer_outputs, i, causal_mask
            )

            # limit possible actions using syntax checker if available
            syntax_mask = None
            eop_syntax_mask = None
            if self.grammar != 'None':
                mask_size = output_logits.shape[1]
                syntax_mask, eop_syntax_mask, grammar_state = self._get_syntax_mask(batch_size, current_tokens,
                                                                                  mask_size, grammar_state)

            # get eop action and new syntax mask if using syntax checker
            if self._two_head:
                eop_preds, eop_output_logits, syntax_mask = self._get_eop_preds(eop_output_logits, eop_syntax_mask,
                                                                              syntax_mask, output_mask_all[:, i])

            # apply softmax
            if syntax_mask is not None:
                assert (output_logits.shape == syntax_mask.shape), '{}:{}'.format(output_logits.shape, syntax_mask.shape)
                output_logits += syntax_mask
                
            if self.setup == 'supervised' or self.setup == 'CEM' or self.setup == 'PPO_option' or self.setup == 'SAC_option':
                preds = self.softmax(output_logits).argmax(dim=-1)
            elif self.setup == 'RL':
                # define distribution over current logits
                dist = FixedCategorical(logits=output_logits)
                # sample actions
                preds = dist.mode().squeeze() if deterministic else dist.sample().squeeze()
                # calculate log probabilities
                if evaluate:
                    assert action[:,i].shape == preds.shape
                    pred_programs_log_probs = dist.log_probs(action[:,i])
                else:
                    pred_programs_log_probs = dist.log_probs(preds)

                if self._two_head:
                    raise NotImplementedError()
                # calculate entropy
                dist_entropy = dist.entropy()
                if self._two_head:
                    raise NotImplementedError()
                pred_programs_log_probs_all.append(pred_programs_log_probs)
                dist_entropy_all.append(dist_entropy.view(-1, 1))
            else:
                raise NotImplementedError()

            # calculate mask for current tokens
            assert preds.shape == output_mask.shape
            if not evaluate:
                if self._two_head:
                    output_mask = (~(eop_preds.to(torch.bool))) * output_mask
                else:
                    output_mask = (~((preds == self.num_program_tokens - 1).to(torch.bool))) * output_mask

                # recalculate first occurrence of <pad> for each program
                first_end_token_idx = torch.min(first_end_token_idx,
                                              ((self.max_program_len * output_mask.float()) +
                                               ((1 - output_mask.float()) * i)).flatten())

            value_all.append(value)
            output_logits_all.append(output_logits)
            pred_programs.append(preds)
            if self._two_head:
                eop_output_logits_all.append(eop_output_logits)
                eop_pred_programs.append(eop_preds)
            if not evaluate:
                output_mask_all[:, i] = output_mask.flatten()

            if self.setup == 'supervised':
                if teacher_enforcing:
                    current_tokens = gt_programs[:, i+1].squeeze()
                else:
                    current_tokens = preds.squeeze()
            else:
                if evaluate:
                    assert self.setup == 'RL'
                    current_tokens = action[:, i]
                else:
                    current_tokens = preds.squeeze()

        # unmask first end-token for two headed policy
        if not evaluate:
            output_mask_all = _unmask_idx(output_mask_all, first_end_token_idx, self.max_program_len).detach()

        # combine all token parameters to get program parameters
        raw_pred_programs_all = torch.stack(pred_programs, dim=1)
        raw_output_logits_all = torch.stack(output_logits_all, dim=1)
        pred_programs_len = torch.sum(output_mask_all, dim=1, keepdim=True)

        if not self._two_head:
            assert output_mask_all.dtype == torch.bool
            pred_programs_all = torch.where(output_mask_all, raw_pred_programs_all,
                                          int(self.num_program_tokens - 1) * torch.ones_like(raw_pred_programs_all))
            eop_pred_programs_all = -1 * torch.ones_like(pred_programs_all)
            raw_eop_output_logits_all = None
        else:
            pred_programs_all = raw_pred_programs_all
            eop_pred_programs_all = torch.stack(eop_pred_programs, dim=1)
            raw_eop_output_logits_all = torch.stack(eop_output_logits_all, dim=1)

        # calculate log_probs, value, actions for program from token values
        if self.setup == 'RL':
            raw_pred_programs_log_probs_all = torch.cat(pred_programs_log_probs_all, dim=1)
            pred_programs_log_probs_all = masked_sum(raw_pred_programs_log_probs_all, output_mask_all,
                                                   dim=1, keepdim=True)

            raw_dist_entropy_all = torch.cat(dist_entropy_all, dim=1)
            dist_entropy_all = masked_mean(raw_dist_entropy_all, output_mask_all, dim=tuple(range(len(output_mask_all.shape))))

            # calculate value for program from token values
            if self.rl_algorithm != 'reinforce':
                if self.value_method == 'mean':
                    raw_value_all = torch.cat(value_all, dim=1)
                    value_all = masked_mean(raw_value_all, output_mask_all, dim=1, keepdim=True)
                else:
                    # calculate value function from hidden states
                    raw_value_all = torch.cat(value_all, dim=1)
                    value_idx = torch.sum(output_mask_all, dim=1, keepdim=True) - 1
                    assert len(value_idx.shape) == 2 and value_idx.shape[1] == 1
                    value_all = torch.gather(raw_value_all, 1, value_idx)

                    # This value calculation is just for sanity check
                    with torch.no_grad():
                        value_idx_2 = first_end_token_idx.clamp(max=self.max_program_len - 1).long().reshape(-1, 1)
                        value_all_2 = torch.gather(raw_value_all, 1, value_idx_2)
                        assert torch.sum(value_all != value_all_2) == 0
                    assert value_all.shape[0] == batch_size
            else:
                value_all = torch.zeros_like(pred_programs_log_probs_all)
        else:
            dist_entropy_all = None
            value_all = None

        return value_all, pred_programs_all, pred_programs_len, pred_programs_log_probs_all, raw_output_logits_all,\
               eop_pred_programs_all, raw_eop_output_logits_all, output_mask_all, dist_entropy_all


class Scalar(nn.Module):
    """
    MLP that produces a scalar multiplier for behavior embeddings.
    
    Takes a behavior embedding b_z with shape (batch_size, latent_dim)
    and outputs a scalar multiplier with shape (batch_size, 1).
    """
    def __init__(self, latent_dim=64, hidden_dim=128):
        super(Scalar, self).__init__()
        
        # Initialize with standard weight initialization
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu')
        )
        
        # Create a simple MLP: latent_dim → hidden_dim → hidden_dim → 1
        self.mlp = nn.Sequential(
            init_(nn.Linear(latent_dim, hidden_dim)),
            nn.ReLU(),
            init_(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            init_(nn.Linear(hidden_dim, 1)),
            # Sigmoid to output values between 0 and 1
            nn.Sigmoid()
        )
        
        # Optional scaling factor to control the range
        self.scale_factor = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        """
        Args:
            x: Behavior embedding tensor of shape (batch_size, latent_dim)
            
        Returns:
            Scalar multiplier of shape (batch_size, 1)
        """
        # Compute scalar multiplier between 0 and scale_factor
        scalar = self.mlp(x) * self.scale_factor
        
        return scalar

class VAE(torch.nn.Module):

    def __init__(self, num_inputs, num_program_tokens, **kwargs):
        super(VAE, self).__init__()
        self._two_head = kwargs['two_head']
        self._vanilla_ae = kwargs['AE']
        self._tanh_after_mu_sigma = kwargs['net']['tanh_after_mu_sigma']
        self._tanh_after_sample   = kwargs['net']['tanh_after_sample']
        self._latent_std_mu       = kwargs['net']['latent_std_mu']
        self._latent_std_sigma    = kwargs['net']['latent_std_sigma']
        self._bz_latent_std_mu = kwargs['net']['bz_latent_std_mu']
        self._bz_latent_std_sigma = kwargs['net']['bz_latent_std_sigma']
        self._latent_mean_pooling = kwargs['net']['latent_mean_pooling']
        self._use_latent_dist     = not kwargs['net']['controller']['use_decoder_dist']
        self._rnn_type            = kwargs['net']['rnn_type']
        self._use_transformer_encoder = kwargs['net']['use_transformer_encoder']
        self._use_transformer_decoder = kwargs['net']['use_transformer_decoder']

        self._use_transformer_encoder_behavior = kwargs['net']['use_transformer_encoder_behavior']
        self._use_transformer_decoder_behavior = kwargs['net']['use_transformer_decoder_behavior']
        
        # For scaling b_z
        self._use_bz_scalar = kwargs['use_bz_scalar']
        if self._use_bz_scalar:
            self.scalar = Scalar()
        
        print("tanh after sample: ", self._tanh_after_sample)
        print("Option VAE latent STD mu:", self._latent_std_mu)
        print("Option VAE latent STD sigma:", self._latent_std_sigma)
        print("Option VAE latent mean pooling:", self._latent_mean_pooling)

        num_outputs = num_inputs

        if kwargs['behavior_representation'] == 'state_sequence':
            self.behavior_encoder = StateBehaviorEncoder(recurrent=kwargs['recurrent_policy'],
                                num_actions=kwargs['dsl']['num_agent_actions'],
                                hidden_size=kwargs['num_lstm_cell_units'], rnn_type=kwargs['net']['rnn_type'],
                                dropout=kwargs['net']['dropout'], use_linear=kwargs['net']['use_linear'],
                                unit_size=kwargs['net']['num_rnn_encoder_units'],
                                **kwargs)
        elif kwargs['behavior_representation'] == 'action_sequence':
            if self._use_transformer_encoder_behavior:
                self.behavior_encoder = ActionBehaviorEncoderTransformer(
                                recurrent=kwargs['recurrent_policy'],
                                num_actions=kwargs['dsl']['num_agent_actions'],
                                hidden_size=kwargs['num_lstm_cell_units'], 
                                rnn_type=kwargs['net']['rnn_type'],
                                dropout=kwargs['net']['dropout'], 
                                use_linear=kwargs['net']['use_linear'],
                                unit_size=kwargs['net']['num_rnn_encoder_units'],
                                transformer_layers=kwargs['net']['transformer_layers'],
                                transformer_heads=kwargs['net']['transformer_heads'],
                                **kwargs)
            else:
                self.behavior_encoder = ActionBehaviorEncoder(recurrent=kwargs['recurrent_policy'],
                                    num_actions=kwargs['dsl']['num_agent_actions'],
                                    hidden_size=kwargs['num_lstm_cell_units'], rnn_type=kwargs['net']['rnn_type'],
                                    dropout=kwargs['net']['dropout'], use_linear=kwargs['net']['use_linear'],
                                    unit_size=kwargs['net']['num_rnn_encoder_units'],
                                    **kwargs)
        if True:
            self.program_encoder = ProgramEncoder(num_inputs, num_outputs, recurrent=kwargs['recurrent_policy'],
                                hidden_size=kwargs['num_lstm_cell_units'], rnn_type=kwargs['net']['rnn_type'],
                                two_head=kwargs['two_head'], dropout=kwargs['net']['dropout'], 
                                use_linear=kwargs['net']['use_linear'], unit_size=kwargs['net']['num_rnn_encoder_units'])

        if True:
            if self._use_transformer_decoder_behavior:
                self.decoder = DecoderTransformer(num_inputs, num_outputs, recurrent=kwargs['recurrent_policy'],
                                    hidden_size=kwargs['num_lstm_cell_units'], rnn_type=kwargs['net']['rnn_type'],
                                    num_program_tokens=num_program_tokens, dropout=kwargs['net']['dropout'], 
                                    unit_size=kwargs['net']['num_rnn_decoder_units'], **kwargs)
            else:
                self.decoder = Decoder(num_inputs, num_outputs, recurrent=kwargs['recurrent_policy'],
                                    hidden_size=kwargs['num_lstm_cell_units'], rnn_type=kwargs['net']['rnn_type'],
                                    num_program_tokens=num_program_tokens, dropout=kwargs['net']['dropout'], 
                                    unit_size=kwargs['net']['num_rnn_decoder_units'], **kwargs)
        self._enc_mu = torch.nn.Linear(kwargs['num_lstm_cell_units'], kwargs['num_lstm_cell_units'])
        self._enc_log_sigma = torch.nn.Linear(kwargs['num_lstm_cell_units'], kwargs['num_lstm_cell_units'])
        self._bz_enc_mu = torch.nn.Linear(kwargs['num_lstm_cell_units'], kwargs['num_lstm_cell_units'])
        self._bz_enc_log_sigma = torch.nn.Linear(kwargs['num_lstm_cell_units'], kwargs['num_lstm_cell_units'])
        self.tanh = torch.nn.Tanh()

    @property
    def latent_dim(self):
        return  self._enc_mu.out_features

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        #std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).to(torch.float).to(h_enc.device)
        std_z = torch.from_numpy(np.random.normal(self._latent_std_mu, self._latent_std_sigma, size=sigma.size())).to(torch.float).to(h_enc.device)
        if self._tanh_after_mu_sigma: #False by default
            mu = self.tanh(mu)
            sigma = self.tanh(sigma)

        self.z_mean = mu
        self.z_sigma = sigma

        return mu, mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def _sample_latent_bz(self, h_enc):
        """
        Return the latent normal sample b_z ~ N(mu, sigma^2)
        """
        mu = self._bz_enc_mu(h_enc)
        log_sigma = self._bz_enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_bz = torch.from_numpy(np.random.normal(self._bz_latent_std_mu, self._bz_latent_std_sigma, size=sigma.size())).to(torch.float).to(h_enc.device)
        if self._tanh_after_mu_sigma: #False by default
            mu = self.tanh(mu)
            sigma = self.tanh(sigma)
        self.b_z_mean = mu
        self.b_z_sigma = sigma
        return mu, mu + sigma * Variable(std_bz, requires_grad=False)  # Reparameterization trick


    @staticmethod
    def latent_loss(z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
        
    @staticmethod
    def combined_latent_loss(z, b_z):
        """
        Combined latent loss that pools the actual z and b_z vectors,
        then calculates KL divergence from the pooled vectors
        """
        # Check that dimensions match
        assert z is not None and b_z is not None, "Actual latent vectors (z and b_z) must be provided"
        assert z.shape == b_z.shape, "Program and behavior vectors must have the same shape"

        # Pool the vectors (concatenate along batch dimension)
        pooled_vectors = torch.cat([z, b_z], dim=0)

        # Calculate combined mean and variance from pooled vectors
        combined_mean = torch.mean(pooled_vectors, dim=0, keepdim=True)
        combined_var = torch.var(pooled_vectors, dim=0, keepdim=True)
        combined_stddev = torch.sqrt(combined_var)

        # Expand to match batch size for calculating KL divergence
        batch_size = z.shape[0]
        combined_mean = combined_mean.repeat(batch_size, 1)
        combined_stddev = combined_stddev.repeat(batch_size, 1)

        # Now compute KL divergence between combined distribution N(combined_mean, combined_stddev²) and N(0,1)
        combined_mean_sq = combined_mean.pow(2)
        combined_stddev_sq = combined_stddev.pow(2)

        # KL(N(μ,σ²) || N
        # (0,1)) = 0.5 * (μ² + σ² - log(σ²) - 1)
        kl_div = 0.5 * torch.mean(combined_mean_sq + combined_stddev_sq - torch.log(combined_stddev_sq) - 1)

        return kl_div

    def forward(self, programs, program_masks, teacher_enforcing, a_h, s_h, a_h_len, s_h_len, deterministic=True):
        program_lens = program_masks.squeeze().sum(dim=-1)

        t = time.time()
        if self._use_transformer_encoder:
            program_masks = program_masks.squeeze().unsqueeze(-2)
            _, h_enc = self.program_encoder(programs, program_masks)
        else:
            _, h_enc = self.program_encoder(programs, program_lens)
        encoder_time = time.time() - t

        if self._latent_mean_pooling:
            output_enc_pad, output_enc_size = pad_packed_sequence(output_enc, batch_first=False)
            output_enc_mean = output_enc_pad.mean(dim=0, keepdim=True)
            assert output_enc_mean.shape == h_enc.data.shape, "output_enc_mean shape {}, h_enc.data shape: {}".format(output_enc_mean.shape, h_enc.data.shape)
            h_enc = output_enc_mean

        if self._rnn_type == 'GRU':
            if self._vanilla_ae:
                z = h_enc.squeezee()
            else:
                z_mu, z = self._sample_latent(h_enc.squeeze())
        elif self._rnn_type == 'LSTM':
            if self._vanilla_ae:
                z = h_enc[0].squeeze()
            else:
                z_mu, z = self._sample_latent(h_enc[0].squeeze())
        else:
            raise NotImplementedError()
        
        pre_tanh_z = z
        emb, beh_mem, beh_mask = self.behavior_encoder(s_h, a_h, s_h_len, a_h_len, return_seq = True)
        b_z_mu, b_z = self._sample_latent_bz(emb) #pretanh behavior embedding
        pre_tanh_b_z = b_z

        if self._tanh_after_sample:
            z_mu = self.tanh(z_mu)
            z = self.tanh(z)
            b_z_mu = self.tanh(b_z_mu)
            b_z = self.tanh(b_z)
        # print(f"z.shape: {z.shape}, b_z.shape: {b_z.shape}")
        
        
        t = time.time()
        
        # programs here is the ground truth
        z_outputs = self.decoder(programs, z, teacher_enforcing=teacher_enforcing, deterministic=deterministic)
        if self._use_bz_scalar:
            # Use the scalar to scale the behavior embedding
            b_z_scalar = self.scalar(b_z)
            b_z = b_z * b_z_scalar
        b_z_outputs = self.decoder(programs, b_z, teacher_enforcing=teacher_enforcing, deterministic=deterministic, behavior_memory=beh_mem, behavior_padding_mask=beh_mask)
         
        # b_z_outputs = self.decoder(programs, b_z, teacher_enforcing=teacher_enforcing, deterministic=deterministic)

        decoder_time = time.time() - t

        return z_outputs, b_z_outputs, z, pre_tanh_z, encoder_time, decoder_time, b_z, pre_tanh_b_z, z_mu, b_z_mu



class ConditionPolicy(NNBase):
    def __init__(self, envs, **kwargs):
        hidden_size = kwargs['num_lstm_cell_units']
        rnn_type = kwargs['net']['rnn_type']
        dropout = kwargs['net']['dropout']
        recurrent = kwargs['recurrent_policy']
        self.num_agent_actions = kwargs['dsl']['num_agent_actions']
        super(ConditionPolicy, self).__init__(recurrent, 2 * hidden_size + self.num_agent_actions, hidden_size, dropout, rnn_type)

        self.envs = envs
        self.state_shape = (kwargs['input_channel'], kwargs['input_height'], kwargs['input_width'])
        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self._rnn_type = rnn_type
        self.max_demo_length = kwargs['max_demo_length']
        self.setup = kwargs['algorithm']
        self.rl_algorithm = kwargs['rl']['algo']['name']
        self.value_method = kwargs['rl']['value_method']
        self.value_embedding = 'eop_rnn'
        self.use_teacher_enforcing =  kwargs['net']['condition']['use_teacher_enforcing']
        self.states_source = kwargs['net']['condition']['observations']

        self._world = Karel_world_supervised(s=None, make_error=False)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.action_encoder = nn.Embedding(self.num_agent_actions, self.num_agent_actions)

        self.state_encoder = nn.Sequential(
            init_(nn.Conv2d(self.state_shape[0], 32, 3, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 4 * 4, hidden_size)), nn.ReLU())

        self.mlp = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                                 init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                                 init_(nn.Linear(hidden_size, self.num_agent_actions)))

        # This check is required only to support backward compatibility to pre-trained models
        if (self.setup =='RL' or self.setup =='supervisedRL') and kwargs['rl']['algo']['name'] != 'reinforce':
            self.critic = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

            self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.softmax = nn.LogSoftmax(dim=-1)

        self.train()


    def _forward_one_pass(self, inputs, rnn_hxs, masks):
        if self.is_recurrent:
            mlp_inputs, rnn_hxs = self._forward_rnn(inputs, rnn_hxs, masks)

        logits = self.mlp(mlp_inputs)

        value = None
        if (self.setup =='RL' or self.setup =='supervisedRL') and self.rl_algorithm != 'reinforce':
            hidden_critic = self.critic(rnn_hxs)
            value = self.critic_linear(hidden_critic)

        return value, logits, rnn_hxs

    def _env_step(self, states, actions, step):
        states = states.detach().cpu().numpy().astype(np.bool_)
        # C x H x W to H x W x C
        states = np.moveaxis(states,[-1,-2,-3], [-2,-3,-1])
        assert states.shape[-1] == self.state_shape[0]
        # karel world expects H x W x C
        if step == 0:
            self._world.reset(states)
        new_states = self._world.step(actions.detach().cpu().numpy())
        new_states = np.moveaxis(new_states,[-1,-2,-3], [-3,-1,-2])
        new_states = torch.tensor(new_states, dtype=torch.float32, device=actions.device)
        return new_states


    def forward(self, s_h, a_h, z, teacher_enforcing=True, eval_actions=None, eval_masks_all=None,
                deterministic=False, evaluate=False):
        """

        :param s_h:
        :param a_h:
        :param z:
        :param teacher_enforcing: True if training in supervised setup or evaluating actions in RL setup
        :param eval_actions:
        :param eval_masks_all:
        :param deterministic:
        :param evaluate: True if setup == RL and evaluating actions, False otherwise
        :return:
        """
        if self.setup == 'supervised':
            assert deterministic == True
        # s_h: B x num_demos_per_program x 1 x C x H x W
        batch_size, num_demos_per_program, demo_len, C, H, W = s_h.shape
        new_batch_size = s_h.shape[0] * s_h.shape[1]
        teacher_enforcing = teacher_enforcing and self.use_teacher_enforcing
        old_states = s_h.squeeze().view(new_batch_size, C, H, W)

        """ get state_embedding of one image per demonstration"""
        new_s_h = s_h[:, :, 0, :, :, :].view(new_batch_size, C, H, W)
        #print(f"condition new_s_h shape: {new_s_h.shape}")
        state_embeddings = self.state_encoder(new_s_h)
        state_embeddings = state_embeddings.view(batch_size, num_demos_per_program, self._hidden_size)
        assert state_embeddings.shape[0] == batch_size and state_embeddings.shape[1] == num_demos_per_program
        state_embeddings = state_embeddings.squeeze()

        """ get intention_embeddings"""
        intention_embedding = z.unsqueeze(1).repeat(1, num_demos_per_program, 1)

        """ get action embeddings for initial actions"""
        actions = (self.num_agent_actions - 1) * torch.ones((batch_size * num_demos_per_program, 1), device=s_h.device,
                                                            dtype=torch.long)

        rnn_hxs = intention_embedding.view(batch_size * num_demos_per_program, self._hidden_size)
        masks = torch.ones((batch_size * num_demos_per_program, 1), device=intention_embedding.device, dtype=torch.bool)
        gru_mask = torch.ones((batch_size * num_demos_per_program, 1), device=intention_embedding.device, dtype=torch.bool)
        assert rnn_hxs.shape[0] == gru_mask.shape[0]
        if self._rnn_type == 'LSTM':
            rnn_hxs = (rnn_hxs, rnn_hxs)
        masks_all = []
        value_all = []
        actions_all = []
        action_logits_all = []
        action_log_probs_all = []
        dist_entropy_all = []
        max_a_h_len = self.max_demo_length-1
        for i in range(self.max_demo_length-1):
            """ get action embeddings and concatenate them with intention and state embeddings """
            action_embeddings = self.action_encoder(actions.view(batch_size, num_demos_per_program))
            inputs = torch.cat((intention_embedding, state_embeddings, action_embeddings), dim=-1)
            inputs = inputs.view(batch_size * num_demos_per_program, -1)

            """ forward pass"""
            value, action_logits, rnn_hxs = self._forward_one_pass(inputs, rnn_hxs, gru_mask)

            """ apply a temporary softmax to get action values to calculate masks """
            if self.setup == 'supervised':
                with torch.no_grad():
                    actions = self.softmax(action_logits).argmax(dim=-1).view(-1, 1)
            elif self.setup == 'RL':
                # define distribution over current logits
                dist = FixedCategorical(logits=action_logits)
                # calculate log probabilities
                if evaluate:
                    assert eval_actions[:, i].shape == actions.squeeze().shape, '{}:{}'.format(eval_actions[:, i].shape,
                                                                                               actions.squeeze().shape)
                    action_log_probs = dist.log_probs(eval_actions[:,i])
                else:
                    # sample actions
                    actions = dist.mode() if deterministic else dist.sample()
                    action_log_probs = dist.log_probs(actions)

                # calculate entropy
                dist_entropy = dist.entropy()
                action_log_probs_all.append(action_log_probs)
                dist_entropy_all.append(dist_entropy.view(-1,1))
            else:
                raise NotImplementedError()

            assert masks.shape == actions.shape
            if not evaluate:
                # NOTE: remove this if check and keep mask update line in case we want to speed up training
                if masks.detach().sum().cpu().item() != 0:
                    masks = masks  * (actions < 5)
                masks_all.append(masks)

            value_all.append(value)
            action_logits_all.append(action_logits)
            actions_all.append(actions)

            """ apply teacher enforcing if ground-truth trajectories available """
            if teacher_enforcing:
                if self.setup == 'supervised':
                    actions = a_h[:, :, i].squeeze().long().view(-1, 1)
                else:
                    actions = eval_actions[:, i].squeeze().long().view(-1, 1)

            """ get the next state embeddings for input to the network"""
            if self.states_source != 'initial_state':
                if teacher_enforcing and self.states_source == 'dataset':
                    new_states = s_h[:, :, i+1, :, :, :].view(s_h.shape[0] * s_h.shape[1], C, H, W)
                else:
                    new_states = self._env_step(old_states, actions, i)
                    assert new_states.shape == (batch_size * num_demos_per_program, C, H, W)

                state_embeddings = self.state_encoder(new_states).view(batch_size, num_demos_per_program,
                                                                         self._hidden_size)
                old_states = new_states

        # unmask first <pad> token
        if not evaluate:
            masks_all = torch.stack(masks_all, dim=1).squeeze()
            first_end_token_idx = torch.sum(masks_all.squeeze(), dim=1)
            _ = list(map(_unmask_idx2, zip(masks_all, first_end_token_idx)))

        action_logits_all = torch.stack(action_logits_all, dim=1)
        assert action_logits_all.shape[-1] == 6

        if self.setup == 'RL':
            masks_all = eval_masks_all if evaluate else masks_all
            actions_all = torch.cat(actions_all, dim=1)

            raw_action_log_probs_all = torch.cat(action_log_probs_all, dim=1)
            action_log_probs_all = masked_sum(raw_action_log_probs_all, masks_all, dim=1, keepdim=True)

            raw_dist_entropy_all = torch.cat(dist_entropy_all, dim=1)
            dist_entropy_all = masked_mean(raw_dist_entropy_all, masks_all, dim=tuple(range(len(masks_all.shape))))

            # calculate value for program from token values
            if self.rl_algorithm != 'reinforce':
                if self.value_method == 'mean':
                    raw_value_all = torch.cat(value_all, dim=1)
                    value_all = masked_mean(raw_value_all, masks_all, dim=1, keepdim=True)
                else:
                    # calculate value function from hidden states
                    raw_value_all = torch.cat(value_all, dim=1)
                    value_idx = torch.sum(masks_all, dim=1, keepdim=True) - 1
                    assert len(value_idx.shape) == 2 and value_idx.shape[1] == 1
                    value_all = torch.gather(raw_value_all, 1, value_idx)

                    # this value calculation is just for sanity check
                    with torch.no_grad():
                        value_idx_2 = first_end_token_idx.clamp(max=self.max_program_len - 1).long().reshape(-1, 1)
                        value_all_2 = torch.gather(raw_value_all, 1, value_idx_2)
                        assert torch.sum(value_all != value_all_2) == 0
                    assert value_all.shape[0] == batch_size
            else:
                value_all = torch.zeros_like(action_log_probs_all)

            value_all = value_all.view(batch_size, num_demos_per_program, 1)
            actions_all = actions_all.view(batch_size, num_demos_per_program, self.max_demo_length - 1)
            masks_all = masks_all.view(batch_size, num_demos_per_program, self.max_demo_length - 1)
            action_log_probs_all = action_log_probs_all.view(batch_size, num_demos_per_program, 1)

        else:
            value_all = None

        return value_all, actions_all, action_log_probs_all, action_logits_all, masks_all, dist_entropy_all


class ProgramVAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ProgramVAE, self).__init__()
        envs = args[0]
        action_space = envs.action_space
        num_outputs = int(action_space.high[0]) if not kwargs['two_head'] else int(action_space.high[0] - 1)
        num_program_tokens = num_outputs if not kwargs['two_head'] else num_outputs + 1
        # two_head policy shouldn't have <pad> token in action distribution, but syntax checker forces it
        # even if its included, <pad> will always have masked probability = 0, so implementation vise it should be fine
        if kwargs['two_head'] and kwargs['grammar'] == 'handwritten':
            num_outputs = int(action_space.high[0])

        self._tanh_after_sample = kwargs['net']['tanh_after_sample']
        self._debug = kwargs['debug']
        self.use_decoder_dist = kwargs['net']['controller']['use_decoder_dist']
        self.use_condition_policy_in_rl = kwargs['rl']['loss']['condition_rl_loss_coef'] > 0.0
        self.num_demo_per_program = kwargs['rl']['envs']['executable']['num_demo_per_program']
        self.max_demo_length = kwargs['rl']['envs']['executable']['max_demo_length']

        self.num_program_tokens = num_program_tokens
        self.teacher_enforcing = kwargs['net']['decoder']['use_teacher_enforcing']
        self.vae = VAE(num_outputs, num_program_tokens, **kwargs)
        self.condition_policy = ConditionPolicy(envs, **kwargs)
        self.POMDP = kwargs['POMDP']

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.vae.latent_dim

    @property
    def is_recurrent(self):
        return self.vae.program_encoder.is_recurrent

    def forward(self, programs, program_masks, s_h_list, a_h, s_h_len, a_h_len, rnn_hxs=None, masks=None, action=None, output_mask_all=None, eop_action=None,
                agent_actions=None, agent_action_masks=None, deterministic=False, evaluate=False):
        #print(f"a_h.shape: {a_h.shape}")
        #print(f"s_h.shape: {init_states.shape}")
        s_h = s_h_list[0]
        s_h_partial = s_h_list[1]
        if self.POMDP:
            encode_sh = s_h_partial
        else:
            encode_sh = s_h
        init_states = s_h[:, :, 0, :, :, :].unsqueeze(2)
        # print(f"init_states.shape: {init_states.shape}")
        if self.vae.decoder.setup == 'supervised':
            z_output, b_z_output, z, pre_tanh_z, encoder_time, decoder_time, b_z, pre_tanh_b_z, z_mu, b_z_mu = self.vae(programs, program_masks, self.teacher_enforcing, deterministic=deterministic, a_h = a_h, s_h = encode_sh, a_h_len = a_h_len, s_h_len = s_h_len)
            _, z_pred_programs, z_pred_programs_len, _, z_output_logits, z_eop_pred_programs, z_eop_output_logits, z_pred_program_masks, _ = z_output
            _, b_z_pred_programs, b_z_pred_programs_len, _, b_z_output_logits, b_z_eop_pred_programs, b_z_eop_output_logits, b_z_pred_program_masks, _ = b_z_output
            _, _, _, z_action_logits, z_action_masks, _ = self.condition_policy(init_states, a_h, z, self.teacher_enforcing,
                                                                         deterministic=deterministic)
            _, _, _, b_z_action_logits, b_z_action_masks, _ = self.condition_policy(init_states, a_h, b_z, self.teacher_enforcing,
                                                                         deterministic=deterministic)
            z_output = {
                'pred_programs': z_pred_programs,
                'pred_programs_len': z_pred_programs_len,
                'output_logits': z_output_logits,
                'eop_pred_programs': z_eop_pred_programs,
                'eop_output_logits': z_eop_output_logits,
                'pred_program_masks': z_pred_program_masks,
                'action_logits': z_action_logits,
                'action_masks': z_action_masks,
                'pre_tanh': pre_tanh_z,
                'z': z,
                'z_mu': z_mu,
            }

            b_z_output = {
                'pred_programs': b_z_pred_programs,
                'pred_programs_len': b_z_pred_programs_len,
                'output_logits': b_z_output_logits,
                'eop_pred_programs': b_z_eop_pred_programs,
                'eop_output_logits': b_z_eop_output_logits,
                'pred_program_masks': b_z_pred_program_masks,
                'action_logits': b_z_action_logits,
                'action_masks': b_z_action_masks,
                'pre_tanh': pre_tanh_b_z,
                'z': b_z,
                'z_mu': b_z_mu,
            }

            return z_output, b_z_output, encoder_time, decoder_time

        # output, z = self.vae(programs, program_masks, self.teacher_enforcing)
        """ VAE forward pass """
        program_lens = program_masks.squeeze().sum(dim=-1)
        t = time.time()
        if self.vae._use_transformer_encoder:
            program_masks = program_masks.squeeze().unsqueeze(-2)
            _, h_enc = self.vae.program_encoder(programs, program_masks)
        else:
            _, h_enc = self.vae.program_encoder(programs, program_lens)
        encoder_time = time.time() - t
        z = h_enc.squeeze() if self.vae._vanilla_ae else self.vae._sample_latent(h_enc.squeeze())
        #print(f"z.shape: {z.shape}")
        pre_tanh_value = None
        if self._tanh_after_sample or not self.use_decoder_dist:
            pre_tanh_value = z
            z = self.vae.tanh(z)

        """ decoder forward pass """
        t = time.time()
        output = self.vae.decoder(programs, z, teacher_enforcing=evaluate, action=action,
                                  output_mask_all=output_mask_all, eop_action=eop_action, deterministic=deterministic,
                                  evaluate=evaluate)
        decoder_time = time.time() - t

        value, pred_programs, pred_programs_len, pred_programs_log_probs, output_logits, eop_pred_programs,\
        eop_output_logits, pred_program_masks, dist_entropy = output

        """ Condition policy rollout using sampled latent vector """
        if self.condition_policy.setup == 'RL' and self.use_condition_policy_in_rl:
            agent_value, agent_actions, agent_action_log_probs, agent_action_logits, agent_action_masks, \
            agent_action_dist_entropy = self.condition_policy(init_states, None, z,
                                                                          teacher_enforcing=evaluate,
                                                                          eval_actions=agent_actions,
                                                                          eval_masks_all=agent_action_masks,
                                                                          deterministic=deterministic,
                                                                          evaluate=evaluate)
        else:
            batch_size = z.shape[0]
            agent_value = torch.zeros((batch_size, self.num_demo_per_program, 1), device=z.device, dtype=torch.long)
            agent_actions = torch.zeros((batch_size, self.num_demo_per_program, self.max_demo_length - 1), device=z.device, dtype=torch.long)
            agent_action_log_probs = torch.zeros((batch_size, self.num_demo_per_program, 1), device=z.device, dtype=torch.float)
            agent_action_masks = torch.zeros((batch_size, self.num_demo_per_program, self.max_demo_length - 1), device=z.device, dtype=torch.bool)
            agent_action_dist_entropy = torch.zeros(1, device=z.device, dtype=torch.float)


        """ calculate latent log probs """
        distribution_params = torch.stack((self.vae.z_mean, self.vae.z_sigma), dim=1)
        if not self.use_decoder_dist:
            latent_log_probs = self.vae.dist.log_probs(z, pre_tanh_value)
            latent_dist_entropy = self.vae.dist.normal.entropy().mean()
        else:
            latent_log_probs, latent_dist_entropy = pred_programs_log_probs, dist_entropy

        return value, pred_programs, pred_programs_log_probs, z, pred_program_masks, eop_pred_programs,\
                agent_value, agent_actions, agent_action_log_probs, agent_action_masks, z, distribution_params, dist_entropy,\
                agent_action_dist_entropy, latent_log_probs, latent_dist_entropy

    def _debug_rl_pipeline(self, debug_input):
        for i, idx in enumerate(debug_input['ids']):
            current_update = "_".join(idx.split('_')[:-1])
            for key in debug_input.keys():
                program_idx = int(idx.split('_')[-1])
                act_program_info = self._debug['act'][current_update][key][program_idx]
                if key == 'ids':
                    assert (act_program_info == debug_input[key][i])
                elif 'agent' in key:
                    assert (act_program_info == debug_input[key].view(-1,act_program_info.shape[0] ,debug_input[key].shape[-1])[i]).all()
                else:
                    assert (act_program_info == debug_input[key][i]).all()

    def act(self, programs, rnn_hxs, masks, s_h, deterministic=False):
        program_masks = programs != self.num_program_tokens-1
        outputs = self(programs.long(), program_masks, s_h, None, rnn_hxs, masks, deterministic=deterministic,
                       evaluate=False)
        return outputs

    def get_value(self, programs, rnn_hxs, masks, s_h, deterministic=False):
        program_masks = programs != self.num_program_tokens-1
        outputs = self(programs.long(), program_masks, s_h, None, rnn_hxs, masks, deterministic=deterministic,
                       evaluate=False)

        value, pred_programs, pred_programs_log_probs, z, pred_program_masks, eop_pred_programs, \
        agent_value, agent_actions, agent_action_log_probs, agent_action_masks, z, distribution_params, dist_entropy, \
        agent_action_dist_entropy, latent_log_probs, latent_dist_entropy = outputs
        return value, agent_value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, output_mask_all, eop_action, agent_actions,
                         agent_action_masks, program_ids, deterministic=False):
        programs, s_h, z = inputs
        program_masks = programs != self.num_program_tokens - 1

        if self._debug:
            self._debug_rl_pipeline({'pred_programs': action,
                                     'pred_program_masks': output_mask_all,
                                     'agent_actions': agent_actions,
                                     'agent_action_masks': agent_action_masks,
                                     'ids': program_ids})

        outputs = self(programs.long(), program_masks, s_h, None, rnn_hxs=rnn_hxs, masks=masks,
                       action=action.long(), output_mask_all=output_mask_all, eop_action=eop_action,
                       agent_actions=agent_actions, agent_action_masks=agent_action_masks,
                       deterministic=deterministic, evaluate=True)
        value, _, pred_programs_log_probs, z, pred_program_masks, _, agent_value, _, agent_action_log_probs, \
        _, _, distribution_params, dist_entropy, agent_action_dist_entropy, latent_log_probs, \
        latent_dist_entropy = outputs

        return value, pred_programs_log_probs, dist_entropy, z, pred_program_masks, agent_value, agent_action_log_probs, \
               agent_action_dist_entropy, distribution_params, latent_log_probs, latent_dist_entropy
