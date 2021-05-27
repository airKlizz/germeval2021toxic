import torch
import torch.nn.functional as F
from transformers import MT5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
model.eval()

text = "it confirms fincher ’s status as a film maker who artfully bends technical know-how to the service of psychological insight"
with torch.no_grad():
    enc = tokenizer(text, return_tensors="pt")
    decoder_input_ids = torch.tensor([tokenizer.pad_token_id]).unsqueeze(0)
    logits = model(**enc, decoder_input_ids=decoder_input_ids)[0]
    print(logits)
    logits = logits.squeeze(1)
    print(logits)
    selected_logits = logits[:, [18205, 47164]]
    print(selected_logits)
    probs = F.softmax(selected_logits, dim=1)
    print(probs)
