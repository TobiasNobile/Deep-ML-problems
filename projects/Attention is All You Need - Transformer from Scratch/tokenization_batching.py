import torch

def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    """
    builds a token-to-id dict with specials first, then corpus tokens in first-seen order.
    """
    mapping = {t:i for i,t in enumerate(specials)}

    counter = len(specials)
    for s in sentences:
        for token in s.split():
            if not token in mapping.keys():
                mapping[token] = counter
                counter += 1

    return mapping

def build_id_to_token_vocab(token_to_id):
    """
    builds the inverse id-to-token dictionary from token_to_id
    """
    return {id:tk for tk, id in token_to_id.items()}

def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    """
    converts whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    """
    return [token_to_id[tk] if tk in token_to_id else token_to_id[unk_token] for tk in sentence.split()]

def decode_ids_to_tokens(ids, id_to_token):
    """
    maps each id in ids to its token string via id_to_token and return the list
    """
    return [id_to_token[i] for i in ids]

def pad_id_sequence(ids, max_len, pad_id):
    """
    returns a list of length exactly max_len, padding with pad_id or truncating.
    """
    if len(ids) < max_len:
        ids.extend([pad_id]*(max_len - len(ids)))
        return ids
    elif len(ids) > max_len:
        return ids[:max_len]
    return ids

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    return torch.tensor(padded_sequences)
