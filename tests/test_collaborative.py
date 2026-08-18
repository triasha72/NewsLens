from newslens.models.collaborative import CollaborativeRecommender, InteractionTriple


def test_bpr_model_learns_basic_preference_signal() -> None:
    triples = [
        InteractionTriple("u1", "a", "b"),
        InteractionTriple("u1", "a", "c"),
        InteractionTriple("u1", "a", "d"),
        InteractionTriple("u2", "b", "a"),
        InteractionTriple("u2", "b", "c"),
        InteractionTriple("u2", "b", "d"),
    ] * 20

    model = CollaborativeRecommender(embedding_dim=16, seed=7).fit(
        triples,
        epochs=25,
        batch_size=16,
        learning_rate=0.03,
    )

    u1 = model.recommend_for_user("u1", candidate_news_ids=["a", "b", "c", "d"], top_k=4)
    assert u1[0].news_id == "a"
    assert model.recommend_for_user("unknown", candidate_news_ids=["a"], top_k=1) == []
