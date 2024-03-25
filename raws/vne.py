#!/usr/bin/env python
# coding: utf-8

# In[15]:


import chess
import torch
from torch.distributions import Categorical as gori

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if not device == 'cuda':
    raise SystemError('cuda not available')

# In[16]:


def t(tensor):
    return torch.tensor(t, device = device, dtype = torch.float16)


# In[17]:


def LMs(board: chess.Board):

    legal_moves = list(board.legal_moves)
    legal_moves_san = [board.san(moves) for moves in legal_moves]
    legal_moves_uci = [board.uci(moves) for moves in legal_moves]

    return legal_moves_uci, legal_moves_san


# In[18]:


def ecode_board(board: chess.Board):

    board_enc = torch.zeros((1, 12, 8, 8), device = device)

    oh_label = {
        'P': 0,
        'R': 1,
        'N': 2,
        'B': 3,
        'Q': 4,
        'K': 5,
        'p': 6,
        'r': 7,
        'n': 8,
        'b': 9,
        'q': 10,
        'k': 11
    }

    for square in chess.SQUARES:

        piece = board.piece_at(square)

        if piece:

            board_count = oh_label[str(piece)]
            row_count, col_count = torch.tensor(divmod(square, 8), device = device)
            col_count = torch.abs(col_count - 7)

            board_enc[0, board_count, row_count, col_count] = 1

    return board_enc


# In[19]:


def ecode_moves(moves: list):

    state = torch.zeros((1, 120, 8, 8), device = device)
    # col_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    col_dict = {'a': 7, 'b': 6, 'c': 5, 'd': 4, 'e': 3, 'f': 2, 'g': 1, 'h': 0}
    for ind, move in enumerate(moves[:120]):

        o_col = torch.tensor(col_dict[move[0]], device = device)
        # o_row = torch.abs(torch.tensor(int(move[1]) - 1, device = device) - 7)
        o_row = torch.tensor(int(move[1]) -1, device = device)
        e_col = torch.tensor(col_dict[move[2]], device = device)
        e_row = torch.tensor(int(move[3]) - 1, device = device)

        state[0, ind, o_row, o_col] = 1

        if move[-1] == 'q':
            state[0, ind, e_row, e_col] = 3
        else:
            state[0, ind, e_row, e_col] = 2

    return state


# In[20]:


def pan_sys(pred_ind, board_state: chess.Board, legal_uci, legal_san, v, captures):


    if pred_ind >= len(legal_uci):
        total_lm_penalty = torch.tensor(0, dtype = torch.float32, device = device)
        total_action_penalty = torch.tensor(0, dtype = torch.float32, device = device)
        move = None
        # v_val = None

    else:
        move = legal_uci[pred_ind]
        actionn = legal_san[pred_ind]
        action = actionn.replace('x', '').replace('+', '').replace('q', '').replace('#', '').replace('=', '')

        file = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        s_piece = ['R', 'N', 'B']
        capture = 'x'


        if 2 < len(action) < 6 and action[0] in file:
            action = f'p{action[1:3]}'
        if len(actionn) == 2:
            action = f'p{actionn}'
        elif len(actionn) >= 6:
            action = f'p{actionn[2:4]}'
        elif (capture in actionn) and (actionn[0] in s_piece) and (actionn[1] in file):
            action = actionn[:1] + actionn[2:]

        col_dict = {'a': t(0), 'b': t(1), 'c': t(2), 'd': t(3), 'e': t(4), 'f': t(5), 'g': t(6), 'h': t(7)}

        penalty_ = {'p': t(2), 'n': t(4), 'b': t(4), 'r': t(7), 'q': t(10), 'k': t(float('inf'))}

        total_lm_penalty = torch.tensor(0.5, dtype = torch.float32, device = device)
        total_action_penalty = torch.tensor(0., dtype = torch.float32, device = device)

        if '#' in actionn:
            total_action_penalty = total_action_penalty + 3

        base_penalty = 0.5

        if (capture not in actionn) and captures:
            total_action_penalty = total_action_penalty - 0
            # v_val = None

        elif  (capture not in actionn) and (not captures):
            total_action_penalty = torch.tensor(0, dtype = torch.float32, device = device)

        else:
            try:
                cap_loc = action[1:3]
                square = (torch.tensor(int(cap_loc[1]), device = device) - 1) * 8 + int(col_dict[cap_loc[0]])
                I_piece = action[0].lower()
                o_piece = str(board_state.piece_at(square.item())).lower()
                refI = penalty_[I_piece]
                refO = penalty_[o_piece]

                if refI < refO:
                    mul = (refO - refI) / 2
                    penalty_score = base_penalty * refO * mul
                elif refI > refO:
                    mul = (refI - refO)
                    penalty_score = base_penalty * (refO / mul)
                elif refI == refO:
                    penalty_score = base_penalty
                else:
                    penalty_score = 0

                total_action_penalty  = total_action_penalty + penalty_score

            except Exception as e:
                total_action_penalty = 'error'

            # v_val = bellman(v)

    return total_action_penalty, total_lm_penalty, move#, v_val


# In[21]:


def pg_loss(prob, r):
    if not prob == 0 or not prob.is_nan():
        loss = -torch.sum(torch.log(prob) * r)
    else:
        loss = None
        
    return loss


# In[22]:


class parse:
    def __init__(self, lr, dom):
        self.loss = t(0)
        self.optim = torch.optim.Adam(lr = lr, params = dom.parameters())
        self.rws = []
        self.probs = []
        
    def dist(self, logi, rw): 
        
        ino = gori(logi)
        choice = ino.sample()
        
        self.probs.append(ino.log_prob(choice))
        self.rws.append(rw)
        
        return choice
                
    def niart(self):
        
        policy_loss = t(0)
        
        for rw, prob in zip(self.rws, self.probs):
            loss = -(rw * prob)
            
            policy_loss = policy_loss + loss
        
        self.optim.zero_grad()
        policy_loss.backward()
        self.optim.step()

