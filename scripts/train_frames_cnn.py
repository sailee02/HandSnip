#!/usr/bin/env python3
import argparse
import json
import os
from typing import Tuple

import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def build_model(num_classes: int, img_size: int) -> Model:
    base = MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights=None)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer=Adam(lr=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_gens(data_root: str, img_size: int, batch: int):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")
    train_aug = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   brightness_range=(0.9, 1.1),
                                   zoom_range=0.1,
                                   horizontal_flip=True)
    plain = ImageDataGenerator(rescale=1./255)
    train_gen = train_aug.flow_from_directory(train_dir, target_size=(img_size, img_size),
                                              batch_size=batch, class_mode="categorical")
    val_gen = plain.flow_from_directory(val_dir, target_size=(img_size, img_size),
                                        batch_size=batch, class_mode="categorical")
    test_gen = plain.flow_from_directory(test_dir, target_size=(img_size, img_size),
                                         batch_size=batch, class_mode="categorical", shuffle=False)
    return train_gen, val_gen, test_gen


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on extracted GIF frames")
    parser.add_argument("--data", type=str, default="dataset/frames_split", help="Root dir with train/val/test")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--img", type=int, default=128)
    parser.add_argument("--out", type=str, default="models/frames_mobilenetv2.h5")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    train_gen, val_gen, test_gen = build_gens(args.data, args.img, args.batch)
    num_classes = len(train_gen.class_indices)
    model = build_model(num_classes, args.img)

    label_map_path = os.path.splitext(args.out)[0] + ".labels.json"
    with open(label_map_path, "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    ckpt = ModelCheckpoint(args.out, monitor="val_accuracy", save_best_only=True, save_weights_only=False)
    es = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)

    model.fit(train_gen,
              epochs=args.epochs,
              validation_data=val_gen,
              callbacks=[ckpt, es, rlrop])

    print("Evaluating best model on test set...")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test accuracy: {test_acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





