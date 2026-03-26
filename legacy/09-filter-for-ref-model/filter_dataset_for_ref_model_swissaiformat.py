import copy
import logging
import argparse

import random
import torch
import numpy as np
import os

import datasets
from transformers import AutoTokenizer

from linearize_swissaiformat import linearise_sample_for_sft

logger = logging.getLogger(__name__)


# To filter out rows that are too long
def add_chat_num_tokens_and_filter_completions(row, tokenizer, max_seq_len):
    """Add chat and len of each completion
    Remove completions which are too long."""
    filtered_conversation_branches = []
    conv_branch_row = copy.deepcopy(row)
    for conv_branch in row["conversation_branches"]:
        conv_branch_row["conversation_branches"] = [conv_branch]
        linear_chat = linearise_sample_for_sft(conv_branch_row)
        chat_tokens = tokenizer.apply_chat_template(linear_chat, tokenize=True)
        # TODO: Assumes we train on only the last message in the chat.
        # TODO: Would be incompatible if we want to train on say (response + verifiable-responses).
        context_tokens = tokenizer.apply_chat_template(
            linear_chat[:-1], tokenize=True, add_generation_prompt=True
        )
        chat_tokens_len = len(chat_tokens)
        context_tokens_len = len(context_tokens)
        if chat_tokens_len <= max_seq_len:
            conv_branch["messages"][-1]["parts"][-1]["metadata"]["chat_num_tokens"] = (
                chat_tokens_len
            )
            conv_branch["messages"][-1]["parts"][-1]["metadata"][
                "context_num_tokens"
            ] = context_tokens_len
            conv_branch["is_reference_completion"] = False
            filtered_conversation_branches.append(conv_branch)

    row["conversation_branches"] = filtered_conversation_branches
    return row


def main(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    data = datasets.load_from_disk(args.dataset)
    # Filter out the completions which prompt+completion length > max_seq_len inside the rows.
    data = data.map(
        add_chat_num_tokens_and_filter_completions,
        num_proc=256,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_len": args.max_seq_len,
        },
    )
    # Filter out the rows which have no completions left.
    data = data.filter(
        lambda conv_branch: len(conv_branch) >= 2,
        input_columns=["conversation_branches"],
        num_proc=256,
    )
    # Record the size of the dataset after filtering
    for split in data.values():
        logger.info(f"Filtered dataset size: {len(split)}")

    logger.info(f"Saving dataset to {args.output_path}")
    data.save_to_disk(args.output_path)
    logger.info("Dataset saved successfully.")
    logger.info("Dataset filtered successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    main(args)
