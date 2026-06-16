"""Extract activations from GPT-2 on gender-marked sentence pairs."""
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Tuple, Dict
import json
import os
from config import SEED, MODEL_NAME, DEVICE, RESULTS_DIR

np.random.seed(SEED)
torch.manual_seed(SEED)

# Gender-marked sentence templates for probe training
# Each template has {subject} placeholder, paired with male/female subjects
TEMPLATES = [
    "{subject} is a doctor who treats patients every day.",
    "{subject} works as an engineer at a tech company.",
    "{subject} is a nurse who cares for patients.",
    "{subject} teaches mathematics at the local school.",
    "{subject} runs a small business in the neighborhood.",
    "{subject} is a lawyer who handles criminal cases.",
    "{subject} works as a chef in a fine dining restaurant.",
    "{subject} is a scientist researching climate change.",
    "{subject} drives a taxi in the city center.",
    "{subject} is a firefighter who saves lives.",
    "{subject} manages a team of software developers.",
    "{subject} is an artist who paints landscapes.",
    "{subject} works as a mechanic at the auto shop.",
    "{subject} is a politician running for office.",
    "{subject} is a student studying biology at university.",
    "{subject} volunteers at the community center on weekends.",
    "{subject} is a pilot who flies international routes.",
    "{subject} works as a journalist covering local news.",
    "{subject} is a farmer who grows organic vegetables.",
    "{subject} is a dentist with a private practice.",
    "{subject} coaches the high school basketball team.",
    "{subject} is a plumber who fixes pipes and drains.",
    "{subject} works as a receptionist at the front desk.",
    "{subject} is a veterinarian who treats injured animals.",
    "{subject} is a construction worker building houses.",
    "{subject} works as a librarian at the public library.",
    "{subject} is a musician who plays in a band.",
    "{subject} is a therapist who helps with mental health.",
    "{subject} works as a cashier at the grocery store.",
    "{subject} is a photographer who takes wedding photos.",
    "{subject} is a police officer patrolling the streets.",
    "{subject} works as a waiter at a busy restaurant.",
    "{subject} is an accountant who manages finances.",
    "{subject} is a hairdresser who styles hair.",
    "{subject} works as a janitor at the school.",
    "{subject} is a social worker helping families.",
    "{subject} is a truck driver making long deliveries.",
    "{subject} works as a secretary in the office.",
    "{subject} is a carpenter who builds furniture.",
    "{subject} is a psychologist conducting research.",
]

MALE_SUBJECTS = [
    "He", "The man", "The boy", "The gentleman", "John",
    "David", "Michael", "James", "Robert", "William",
    "The father", "The husband", "The brother", "The uncle", "The grandfather",
    "The king", "The prince", "The nephew", "The son", "My dad",
]

FEMALE_SUBJECTS = [
    "She", "The woman", "The girl", "The lady", "Mary",
    "Sarah", "Jennifer", "Emily", "Jessica", "Elizabeth",
    "The mother", "The wife", "The sister", "The aunt", "The grandmother",
    "The queen", "The princess", "The niece", "The daughter", "My mom",
]


def generate_gendered_sentences(n_per_gender: int = 500) -> Tuple[List[str], np.ndarray]:
    """Generate gender-marked sentences and labels (0=male, 1=female)."""
    sentences = []
    labels = []

    for i in range(n_per_gender):
        template = TEMPLATES[i % len(TEMPLATES)]
        male_subj = MALE_SUBJECTS[i % len(MALE_SUBJECTS)]
        female_subj = FEMALE_SUBJECTS[i % len(FEMALE_SUBJECTS)]

        sentences.append(template.format(subject=male_subj))
        labels.append(0)
        sentences.append(template.format(subject=female_subj))
        labels.append(1)

    return sentences, np.array(labels)


def load_model():
    """Load GPT-2 model and tokenizer."""
    print(f"Loading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model.eval()
    model.to(DEVICE)
    print(f"Model loaded. Layers: {model.config.n_layer}, Hidden dim: {model.config.n_embd}")
    return model, tokenizer


def extract_activations(model, tokenizer, sentences: List[str],
                        batch_size: int = 16) -> Dict[int, np.ndarray]:
    """Extract residual stream activations at each layer for each sentence.

    Returns dict mapping layer_idx -> array of shape (n_sentences, hidden_dim).
    Uses mean pooling over token positions.
    """
    n_layers = model.config.n_layer
    all_activations = {i: [] for i in range(n_layers + 1)}  # +1 for embedding layer

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=64)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)
        attention_mask = inputs["attention_mask"]  # (batch, seq_len)

        for layer_idx, hs in enumerate(hidden_states):
            # Mean pool over non-padding tokens
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1)  # (batch, hidden_dim)
            all_activations[layer_idx].append(pooled.cpu().numpy())

        if (start // batch_size) % 10 == 0:
            print(f"  Processed {start + len(batch)}/{len(sentences)} sentences")

    # Concatenate batches
    for layer_idx in all_activations:
        all_activations[layer_idx] = np.concatenate(all_activations[layer_idx], axis=0)

    return all_activations


def extract_and_save(n_per_gender: int = 500):
    """Main function: generate sentences, extract activations, save."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sentences, labels = generate_gendered_sentences(n_per_gender)
    print(f"Generated {len(sentences)} sentences ({n_per_gender} per gender)")

    model, tokenizer = load_model()

    print("Extracting activations...")
    activations = extract_activations(model, tokenizer, sentences)

    # Save
    np.save(os.path.join(RESULTS_DIR, "activations.npy"),
            {k: v for k, v in activations.items()}, allow_pickle=True)
    np.save(os.path.join(RESULTS_DIR, "labels.npy"), labels)

    # Save sentences for reference
    with open(os.path.join(RESULTS_DIR, "sentences.json"), "w") as f:
        json.dump(sentences, f, indent=2)

    print(f"Saved activations for {len(activations)} layers, "
          f"shape per layer: {activations[0].shape}")

    return model, tokenizer, activations, labels, sentences


if __name__ == "__main__":
    extract_and_save()
