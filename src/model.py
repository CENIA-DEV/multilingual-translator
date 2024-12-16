import torch
import torch.nn as nn
import torch.nn.functional as F
from accmt import AcceleratorModule
from arguments import TrainingArguments
from transformers import AutoModelForSeq2SeqLM

class Translator(AcceleratorModule):
    def __init__(self, args: TrainingArguments, config, tokenizer):
        self.args = args
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            args.MODEL, cache_dir="cache", config=config
        )

        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        
        self.tokenizer = tokenizer
        self.vocab_size = config.vocab_size
        self.alpha = args.RDROP_ALPHA
        
    def compute_kl_loss(self, p, q, pad_mask=None):    
        p_loss = F.kl_div(
            F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none'
        )
        q_loss = F.kl_div(
            F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none'
        )
        
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        p_loss = p_loss.mean()
        q_loss = q_loss.mean()

        loss = (p_loss + q_loss) / 2
        return loss
    
    def compute_loss(self, logits, labels, label_smoothing=0.0):
        return F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            label_smoothing=label_smoothing
        )
    
    def training_step(self, batch):
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        compute_loss_kwargs = {
            "labels": batch["labels"], "label_smoothing": self.args.LABEL_SMOOTHING
        }
        if self.args.RDROP:
            output1 = self.model(**batch)
            output2 = self.model(**batch)

            if self.args.LABEL_SMOOTHING is None:
                output1_loss = output1.loss
                output2_loss = output2.loss
            else:
                output1_loss = self.compute_loss(output1.logits, **compute_loss_kwargs)
                output2_loss = self.compute_loss(output2.logits, **compute_loss_kwargs)

            nll_loss = 0.5 * (output1_loss + output2_loss)
            kl_loss = self.compute_kl_loss(output1.logits, output2.logits)
            
            loss = nll_loss + self.alpha * kl_loss

            return loss

        output = self.model(**batch)
        if not self.args.LABEL_SMOOTHING:
            return output.loss
        
        smoothed_loss = self.compute_loss(output.logits, **compute_loss_kwargs)
        return smoothed_loss
    
    def validation_step(self, batch):
        directions = batch
        return_dict = dict()
        losses = []
        for direction, inputs in directions.items():
            labels = inputs["labels"].clone()
            inputs["labels"][inputs["labels"] == self.tokenizer.pad_token_id] = -100
            outputs = self.model(**inputs)
            losses.append(outputs.loss)
            output_token_ids = torch.argmax(outputs.logits, dim=-1)
            return_dict[f"bleu_{direction}"] = (output_token_ids, labels)

        avg_loss = torch.mean(torch.stack(losses))
        return_dict["loss"] = avg_loss

        return return_dict
