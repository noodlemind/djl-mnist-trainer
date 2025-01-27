package io.github.noodlemind.djlmnisttrainer;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public final class MnistApplication {
	private static final Logger logger = LoggerFactory.getLogger(MnistApplication.class);
	private static final Path MODEL_DIR = Paths.get("build/model");
	private static final String MODEL_NAME = "mnist";
	private static final String PARAMS_FILE = MODEL_NAME + "-0002.params";

	public static void main(String[] args) {
		try {
			train();
		} catch (Exception e) {
			logger.error("Training failed: {}", e.getMessage(), e);
			System.exit(1);
		}
	}

	private static void train() throws IOException, TranslateException {
		Files.createDirectories(MODEL_DIR);
		logger.info("Using model directory: {}", MODEL_DIR.toAbsolutePath());

		try (Model model = Model.newInstance(MODEL_NAME)) {
			model.setBlock(new Mlp(
					Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH,
					Mnist.NUM_CLASSES,
					new int[]{128, 64}));

			RandomAccessDataset trainingSet = getDataset(Dataset.Usage.TRAIN);
			RandomAccessDataset validateSet = getDataset(Dataset.Usage.TEST);

			DefaultTrainingConfig config = setupTrainingConfig();

			try (Trainer trainer = model.newTrainer(config)) {
				trainer.setMetrics(new Metrics());
				trainer.initialize(new Shape(1, Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH));

				logger.info("Training MNIST model...");
				EasyTrain.fit(trainer, 2, trainingSet, validateSet);

				// Save model with properties
				model.setProperty("Epoch", "2");
				model.setProperty("Architecture", "MLP");
				model.setProperty("Engine", "PyTorch");
				model.save(MODEL_DIR, MODEL_NAME);

				// Verify the saved model params file
				Path paramsFile = MODEL_DIR.resolve(PARAMS_FILE);
				if (Files.exists(paramsFile)) {
					logger.info("Model parameters saved to: {}", paramsFile);
					logger.info("Model size: {} bytes", Files.size(paramsFile));
				} else {
					throw new IOException("Model parameters file was not created: " + paramsFile);
				}
			}
		}
	}

	private static DefaultTrainingConfig setupTrainingConfig() {
		SaveModelTrainingListener listener = new SaveModelTrainingListener(MODEL_DIR.toString());
		listener.setSaveModelCallback(
				trainer -> {
					try {
						TrainingResult result = trainer.getTrainingResult();
						Model model = trainer.getModel();
						float accuracy = result.getValidateEvaluation("Accuracy");
						model.setProperty("Accuracy", String.format("%.5f", accuracy));
						model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
					} catch (Exception e) {
						logger.error("Failed to save model callback", e);
					}
				});

		return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
				       .addEvaluator(new Accuracy())
				       .addTrainingListeners(TrainingListener.Defaults.logging(MODEL_DIR.toString()))
				       .addTrainingListeners(listener);
	}

	private static RandomAccessDataset getDataset(Dataset.Usage usage) throws IOException {
		var mnist = Mnist.builder()
				            .optUsage(usage)
				            .setSampling(32, true)
				            .build();
		mnist.prepare(new ProgressBar());
		return mnist;
	}
}