import numpy as np
import torch

from encoding import BOARD_SQUARES, board_square_tokens, board_to_input, child_board


@torch.inference_mode()
def batched_policy_step(boards, model, device, temperature=0.0):
    board_inputs = torch.tensor(
        [board_to_input(board) for board in boards],
        dtype=torch.long,
        device=device,
    )
    board_logits, value = model(board_inputs)
    log_probs = torch.log_softmax(board_logits, dim=-1).float().cpu().numpy()
    values = value.cpu().tolist()

    moves = []
    for b, board in enumerate(boards):
        candidates = list(board.legal_moves)
        child_tokens = np.array(
            [board_square_tokens(child_board(board, move)) for move in candidates],
            dtype=np.int64,
        )
        scores = log_probs[b][np.arange(BOARD_SQUARES)[None, :], child_tokens].sum(
            axis=-1
        )
        if temperature <= 0:
            choice = int(scores.argmax())
        else:
            probs = np.exp((scores - scores.max()) / temperature)
            choice = int(np.random.choice(len(candidates), p=probs / probs.sum()))
        moves.append(candidates[choice])

    return moves, values
