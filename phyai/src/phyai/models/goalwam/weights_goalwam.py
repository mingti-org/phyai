"""Map GoalWAM inference checkpoint names to PhyAI parameter names."""


def goalwam_weight_remap(key: str) -> str | None:
    """Keep unknown names visible to the strict checkpoint loader."""
    if key == "language_model.lm_head.weight":
        return None
    prefix = "language_model.model.layers."
    if key.startswith(prefix):
        index, leaf = key[len(prefix) :].split(".", 1)
        generator = "_moe_gen" in leaf
        leaf = leaf.replace("_moe_gen", "")
        attention = "cross_attention" if generator else "self_attn"
        for source, target in (
            ("self_attn.q_proj.", f"{attention}.to_q."),
            ("self_attn.k_proj.", f"{attention}.to_k."),
            ("self_attn.v_proj.", f"{attention}.to_v."),
            ("self_attn.o_proj.", f"{attention}.to_out."),
            ("self_attn.q_norm.", f"{attention}.norm_q."),
            ("self_attn.k_norm.", f"{attention}.norm_k."),
        ):
            if leaf.startswith(source):
                leaf = target + leaf[len(source) :]
                break
        tower = "gen_layers" if generator else "language_model.layers"
        return f"{tower}.{index}.{leaf}"
    for source, target in (
        ("language_model.model.norm_moe_gen.", "norm_moe_gen."),
        ("language_model.model.embed_tokens.", "language_model.embed_tokens."),
        ("language_model.model.norm.", "language_model.norm."),
        ("time_embedder.mlp.0.", "time_embedder.linear_1."),
        ("time_embedder.mlp.2.", "time_embedder.linear_2."),
        ("vae2llm.", "proj_in."),
        ("llm2vae.", "proj_out."),
        ("action2llm.", "action_proj_in."),
        ("llm2action.", "action_proj_out."),
    ):
        if key.startswith(source):
            return target + key[len(source) :]
    return key
