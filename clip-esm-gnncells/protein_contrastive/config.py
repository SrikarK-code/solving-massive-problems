# from types import SimpleNamespace

# config = SimpleNamespace(
#     batch_size=32,
#     learning_rate=1e-4,
#     queue_size=32 * 128,
#     latent_dim=1024,
#     hidden_dim=512,
#     epochs=20,
#     temperature_init=0.07,
#     momentum=0.999,
#     early_stop_patience=5,
#     gradient_clip_norm=5.0,
#     optimizer="adamw",
#     scheduler="cosine",
#     viz_frequency=5,
#     loss_weights={
#         'gmc': 1.0,
#         'contrastive': 1.0,
#         'barlow': 0.5,
#         'vicreg': 0.5,
#         'wasserstein': 0.1,
#         'hyperbolic': 0.3
#     },
#     projection_dim=512,
#     shared_dim=1024,
#     hard_negative_mining=True,
#     min_neg_samples=5,
#     max_neg_samples=4096,
#     threshold_factor=0.1
# )



from types import SimpleNamespace

config = SimpleNamespace(
    batch_size=32,
    learning_rate=1e-4,
    queue_size=32 * 128,
    latent_dim=1024,
    hidden_dim=512,
    epochs=20,
    temperature_init=0.07,
    momentum=0.999,
    early_stop_patience=5,
    gradient_clip_norm=5.0,
    optimizer="adamw",
    scheduler="cosine",
    viz_frequency=5,
    loss_weights={
        'gmc': 1.0,
        'contrastive': 1.0,
        'barlow': 0.5,
        'vicreg': 0.5,
        'wasserstein': 0.1,
        'hyperbolic': 0.3
    },
    projection_dim=512,
    shared_dim=1024,
    hard_negative_mining=True,
    min_neg_samples=5,
    max_neg_samples=4096,
    threshold_factor=0.1,
    esm_dim=1280,
    dpt_dim=15,
    gene_dim=11831,
    n_heads=8,
    n_layers=3,
    n_neighbors=32,
    dropout = 0.1,
    temperature_decay = 0.99,
    temperature_min = 0.05,
    use_time_encoding = True
)
