from langgraph.graph import StateGraph, END
from .graph_state import PipelineState
from .graph_nodes import make_nodes

def build_pipeline_graph(config, llm_generator, profile_results,
                         all_dictionaries, minhash_results, dataset_summaries):

    (
        node_generate_rules,
        node_assess_rules,
        node_regenerate_rules,
        fan_out_validation,
        node_validate_table,
        node_inspect_rules,
        node_revise_rules,
        route_after_assessment,
        route_after_inspection,
    ) = make_nodes(config, llm_generator, profile_results,
                   all_dictionaries, minhash_results, dataset_summaries)

    g = StateGraph(PipelineState)
    g.add_node("generate_rules",   node_generate_rules)
    g.add_node("assess_rules",     node_assess_rules)
    g.add_node("regenerate_rules", node_regenerate_rules)
    g.add_node("validate_table",   node_validate_table)
    g.add_node("inspect_rules",    node_inspect_rules)
    g.add_node("revise_rules",     node_revise_rules)

    g.set_entry_point("generate_rules")
    g.add_edge("generate_rules", "assess_rules")
    g.add_conditional_edges(
        "assess_rules",
        route_after_assessment,
        ["validate_table", "regenerate_rules"]
    )
    g.add_conditional_edges("regenerate_rules", fan_out_validation, ["validate_table"])
    g.add_edge("validate_table", "inspect_rules")
    g.add_conditional_edges(
        "inspect_rules",
        route_after_inspection,
        {"revise_rules": "revise_rules", "regenerate_rules": "regenerate_rules", END: END}
    )
    g.add_conditional_edges("revise_rules", fan_out_validation, ["validate_table"])

    return g.compile()