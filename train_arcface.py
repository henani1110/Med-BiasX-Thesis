import pickle
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import utils.config as config
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable
from tensorboardX import SummaryWriter

# writer_tsne = SummaryWriter('runs/tsne')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
EPS = 1e-8
KL_DENOM_EPS = 1e-4
EXP_CLAMP = 10.0

def compute_supcon_loss(feats, qtype):
    tau = 1.0
    if torch.is_tensor(qtype):
        qtype = qtype.to(DEVICE).long()
    else:
        mapped = []
        idx_map = {}
        next_idx = 0
        for item in qtype:
            if item not in idx_map:
                idx_map[item] = next_idx
                next_idx += 1
            mapped.append(idx_map[item])
        qtype = torch.tensor(mapped, device=DEVICE, dtype=torch.long)
    feats_filt = F.normalize(feats, dim=1)
    logits = torch.matmul(feats_filt, feats_filt.T) / tau
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    bsz = qtype.size(0)
    logits_mask = torch.ones((bsz, bsz), device=DEVICE) - torch.eye(bsz, device=DEVICE)
    pos_mask = (qtype.reshape(-1, 1) == qtype.reshape(1, -1)).float() * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(EPS))

    pos_count = pos_mask.sum(dim=1).clamp_min(1.0)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count
    sup_con_loss = -mean_log_prob_pos.mean()
    return torch.nan_to_num(sup_con_loss, nan=0.0, posinf=1e4, neginf=0.0)

def compute_acc(logits, labels):
    pred = torch.argmax(logits, dim = 1)
    pred = pred.detach().cpu().numpy()
    score = (pred == np.array(labels))
    tot_correct = score.sum()
    return tot_correct


def compute_score_with_logits(logits, labels):
    _, log_index = logits.max(dim=1, keepdim=True)
    scores = labels.gather(dim=1, index=log_index)
    return scores
    
def compute_loss(output, labels):

    #Function for calculating loss
    
    ce_loss = nn.CrossEntropyLoss(reduction='mean')(output, labels.squeeze(-1).long())
    
    return ce_loss


def saved_for_eval(dataloader, results, question_ids, answer_preds):
    """ Save as a format accepted by the evaluation server. """
    _, answer_ids = answer_preds.max(dim=1)
    answers = [dataloader.dataset.label2ans[i] for i in answer_ids]
    for q, a in zip(question_ids, answers):
        entry = {
            'question_id': q.item(),
            'answer': a,
        }
        results.append(entry)
    return results


def train(model, m_model, optim, train_loader, loss_fn, tracker, writer, tb_count, epoch, args):

    loader = tqdm(train_loader, ncols=0)
    loss_trk = tracker.track('loss', tracker.MovingMeanMonitor(momentum=0.99))
    acc_trk = tracker.track('acc', tracker.MovingMeanMonitor(momentum=0.99))
    warmup_epochs = max(0, int(getattr(args, 'aux_warmup_epochs', 0)))
    aux_max_weight = float(getattr(args, 'aux_loss_weight', 1.0))
    if warmup_epochs <= 0:
        aux_scale = aux_max_weight
    else:
        aux_scale = min(1.0, float(epoch + 1) / float(warmup_epochs)) * aux_max_weight

    for v, q, a, mg, q_id, f1, type, a_type in loader:
        v = v.to(DEVICE)
        q = q.to(DEVICE)
        a = a.to(DEVICE)
        mg = mg.to(DEVICE)
        f1 = f1.to(DEVICE)
        gt = torch.argmax(a, 1)

        ans = []
        ans_tokens = []
        ans_index = torch.argmax(a, dim=1, keepdim=True).data.cpu()
        for index in ans_index:
            ans.append(train_loader.dataset.label2ans[index])
        for w in ans:
            if w not in train_loader.dataset.dictionary.word2idx:
                ans_tokens.append(18455)
            else:
                ans_tokens.append(train_loader.dataset.dictionary.word2idx[w])
        ans_tokens = torch.from_numpy(np.array(ans_tokens))
        ans_tokens = Variable(ans_tokens).to(DEVICE)

        hidden_, ce_logits, q_logits, a_logits = model(v, q, ans_tokens)
        hidden, pred = m_model(hidden_, ce_logits, mg, epoch, a)

        dict_args = {'margin': mg, 'hidden': hidden, 'epoch': epoch, 'per': f1}

        #If bias-injection or learnable margins is enabled.
        if config.learnable_margins or config.bias_inject:
            #Use cross entropy loss to train the bias-injecting module
            ce_loss = - F.log_softmax(ce_logits, dim=-1) * a
            ce_loss = ce_loss * f1
            loss = ce_loss.sum(dim=-1).mean() + loss_fn(hidden, a, **dict_args)
        else:
            loss = loss_fn(hidden, a, **dict_args)
        
        aux_total = torch.tensor(0.0, device=DEVICE)

        # Add the supcon loss, as mentioned in Section 3 of main paper.
        if config.supcon:
            aux_total = aux_total + compute_supcon_loss(hidden_, gt)

        a_logits_safe = torch.clamp(a_logits, -30.0, 30.0)
        ce_logits_safe = torch.clamp(ce_logits, -30.0, 30.0)
        q_logits_safe = torch.clamp(q_logits, -30.0, 30.0)

        # DDC (KL-based alignment) can be disabled for ablation via --no-ddc.
        if not getattr(args, 'no_ddc', False):
            kl1 = F.kl_div(
                F.log_softmax(a_logits_safe, dim=-1),
                F.softmax(ce_logits_safe, dim=-1),
                reduction='batchmean'
            )
            kl2 = F.kl_div(
                F.log_softmax(a_logits_safe, dim=-1),
                F.softmax(q_logits_safe, dim=-1),
                reduction='batchmean'
            )

            kl2_safe = kl2.clamp_min(KL_DENOM_EPS)
            bias = torch.log1p((kl1 / kl2_safe).clamp(max=1e4))
            direction = torch.exp((kl1 - kl2).clamp(min=-EXP_CLAMP, max=EXP_CLAMP))
            direction = torch.where(kl1 > kl2, torch.zeros_like(direction), direction)
            kl_loss = torch.nan_to_num(direction + bias, nan=0.0, posinf=1e4, neginf=0.0)
            kl_loss = kl_loss.clamp(max=50.0)
            aux_total = aux_total + kl_loss
        else:
            kl1 = torch.tensor(0.0, device=DEVICE)
            kl2 = torch.tensor(0.0, device=DEVICE)

        # ECC (energy constraint) can be disabled for ablation via --no-ecc.
        if not getattr(args, 'no_ecc', False):
            Ec_joint = torch.logsumexp(ce_logits_safe, dim=1)
            Ec_q = torch.logsumexp(q_logits_safe, dim=1)
            En = torch.pow(F.relu(args.m + Ec_joint - Ec_q), 2).mean()
            En = torch.nan_to_num(En, nan=0.0, posinf=1e4, neginf=0.0).clamp(max=50.0)
            aux_total = aux_total + En

        scaled_aux = aux_scale * aux_total
        max_aux_loss = float(getattr(args, 'max_aux_loss', 0.0))
        if max_aux_loss > 0:
            scaled_aux = scaled_aux.clamp(max=max_aux_loss)

        loss = loss + scaled_aux

        if not torch.isfinite(loss):
            print('[warn] non-finite loss detected. skip batch.')
            print('ce_logits finite:', torch.isfinite(ce_logits).all().item())
            print('q_logits finite:', torch.isfinite(q_logits).all().item())
            print('a_logits finite:', torch.isfinite(a_logits).all().item())
            print('kl1:', float(kl1.detach().cpu()), 'kl2:', float(kl2.detach().cpu()))
            optim.zero_grad()
            continue
        
        writer.add_scalars('data/train', {
            'loss': float(loss.item())
        }, tb_count)
        tb_count += 1

        loss.backward()

        all_params = list(model.parameters()) + list(m_model.parameters())
        nn.utils.clip_grad_norm_(all_params, 0.25)

        grads_finite = True
        for p in all_params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grads_finite = False
                break

        if not grads_finite:
            print('[warn] non-finite gradients detected. skip optimizer step.')
            optim.zero_grad()
            continue

        optim.step()
        optim.zero_grad()
        
        # Ensemble the logit heads, as mentioned in Section 3 of the main paper, if bias-injection is enabled
        if config.bias_inject or config.learnable_margins:
          ce_logits = F.normalize(ce_logits)
          pred_l = F.normalize(pred)
          pred = (ce_logits + pred_l) / 2
        batch_score = compute_score_with_logits(pred, a.data)

        fmt = '{:.4f}'.format
        loss_trk.append(loss.item())
        acc_trk.append(batch_score.mean())
        loader.set_postfix(loss=fmt(loss_trk.mean.value),
                           acc=fmt(acc_trk.mean.value))

    train_loss = float(loss_trk.mean.value) if loss_trk.mean.value is not None else float('nan')
    train_acc = float(acc_trk.mean.value) if acc_trk.mean.value is not None else float('nan')
    return tb_count, train_loss, train_acc


#Evaluation code
def evaluate(model, m_model, dataloader, epoch=0, write=False):
    score = 0
    results = []  # saving for evaluation
    type_score = 0
    qat_score = {}
    qat_total = {}
    for v, q, a, mg, q_id, f1, qtype, a_type in tqdm(dataloader, ncols=0, leave=True):
        v = v.to(DEVICE)
        q = q.to(DEVICE)
        mg = mg.to(DEVICE)
        a = a.to(DEVICE)
        hidden_, ce_logits, q_logits, _ = model(v, q, None)
        hidden, pred = m_model(hidden_, ce_logits, mg, epoch, a)
        
        #Ensemble the logit heads
        if config.learnable_margins or config.bias_inject:
          ce_logits = F.softmax(F.normalize(ce_logits) / config.temp, 1)
          pred_l = F.softmax(F.normalize(pred), 1)
          pred = config.alpha * pred_l + (1-config.alpha) * ce_logits

        if write:
            results = saved_for_eval(dataloader, results, q_id, pred)
        batch_score = compute_score_with_logits(pred, a).sum(1)
        score += batch_score.sum()

        for i in range(len(batch_score)):
            q_key = str(qtype[i])
            a_key = str(a_type[i])
            qat_score[q_key] = qat_score.get(q_key, 0) + batch_score[i]
            qat_total[q_key] = qat_total.get(q_key, 0) + 1
            qat_score[a_key] = qat_score.get(a_key, 0) + batch_score[i]
            qat_total[a_key] = qat_total.get(a_key, 0) + 1  
        
    print(score, len(dataloader.dataset))
    score = score / len(dataloader.dataset)

    for key in qat_score:
        print(str(key) + ": " + str((qat_score[key]/qat_total[key]*100).item()))
    
    # if write:
    #     print("saving prediction results to disk...")
    #     result_file = 'vqa_{}_{}_{}_{}_results.json'.format(
    #         config.task, config.test_split, config.version, epoch)
        # with open(result_file, 'w') as fd:
            # json.dump(results, fd)
    print(score)
    return score
