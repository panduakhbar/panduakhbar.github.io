from rfdetr import RFDETRNano

model = RFDETRNano()
model.train(
    dataset_dir="/content/palmtree_detection",
    output_dir="/content/rfdetr_nano_output",
    epochs=30,
    resolution=384,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    early_stopping=True,
    early_stopping_patience=5,
    early_stopping_min_delta=0.001,
    skip_best_epochs=3,
    # Validasi tidak perlu setiap epoch
    eval_interval=5,
    fp16_eval=True,
    compute_val_loss=False,
    checkpoint_interval=5,
    use_ema=True,
    tensorboard=False,
    wandb=False,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
