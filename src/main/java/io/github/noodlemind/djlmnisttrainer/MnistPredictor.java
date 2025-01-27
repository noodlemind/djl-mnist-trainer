package io.github.noodlemind.djlmnisttrainer;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import io.github.noodlemind.djlmnisttrainer.translator.MnistTranslator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public final class MnistPredictor {
	private static final Logger logger = LoggerFactory.getLogger(MnistPredictor.class);
	private static final Path MODEL_DIR = Paths.get("build/model");
	private static final String MODEL_NAME = "mnist";
	private static final String PARAMS_FILE = MODEL_NAME + "-0002.params";
	private static final int TOP_K_PREDICTIONS = 3;
	private static final int SEPARATOR_LENGTH = 80;

	public static Classifications predict(String imagePath) throws IOException, TranslateException, MalformedModelException {
		validateInput(imagePath);
		Path imageFile = Paths.get(imagePath);

		logStartPrediction(imageFile);
		Path paramsPath = validateAndGetModelPath();

		try (Model model = Model.newInstance(MODEL_NAME)) {
			configureModel(model);
			model.load(MODEL_DIR, MODEL_NAME);

			Translator<Image, Classifications> translator = new MnistTranslator();

			try (Predictor<Image, Classifications> predictor = model.newPredictor(translator)) {
				Image img = ImageFactory.getInstance().fromFile(imageFile);
				logImageProcessing(img);

				Classifications result = predictor.predict(img);
				validateAndLogResults(result);
				return result;
			}
		}
	}

	private static void logStartPrediction(Path imageFile) {
		logger.info("\n" + "=".repeat(SEPARATOR_LENGTH));
		logger.info("Starting MNIST Prediction");
		logger.info("=".repeat(SEPARATOR_LENGTH));
		logger.info("Input Image Details:");
		logger.info("└── Path: {}", imageFile.toAbsolutePath());
	}

	private static Path validateAndGetModelPath() throws IOException {
		Path paramsPath = MODEL_DIR.resolve(PARAMS_FILE);
		validateModel(paramsPath);
		logger.info("\nModel Information:");
		logger.info("└── Parameters: {}", paramsPath);
		logger.info("└── Size: {} bytes", Files.size(paramsPath));
		return paramsPath;
	}

	private static void validateInput(String imagePath) {
		if (imagePath == null || imagePath.trim().isEmpty()) {
			throw new IllegalArgumentException("Image path cannot be null or empty");
		}
		if (!Files.exists(Paths.get(imagePath))) {
			throw new IllegalArgumentException("Image file does not exist: " + imagePath);
		}
	}

	private static void validateModel(Path paramsPath) throws IOException {
		if (!Files.exists(paramsPath)) {
			throw new IOException("Model parameters not found at: " + paramsPath);
		}
	}

	private static void configureModel(Model model) {
		model.setBlock(new Mlp(Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH, Mnist.NUM_CLASSES, new int[]{128, 64}));

		logger.info("\nLoading model configuration:");
		logger.info("└── Model Name: {}", MODEL_NAME);
		logger.info("└── Input Size: {}x{}", Mnist.IMAGE_WIDTH, Mnist.IMAGE_HEIGHT);
		logger.info("└── Output Classes: {}", Mnist.NUM_CLASSES);
	}

	private static void logImageProcessing(Image img) {
		logger.info("\nProcessing Image:");
		logger.info("└── Original size: {}x{}", img.getWidth(), img.getHeight());
		logger.info("└── Converting to: {}x{}", Mnist.IMAGE_WIDTH, Mnist.IMAGE_HEIGHT);
	}

	private static void validateAndLogResults(Classifications result) throws TranslateException {
		if (result == null || result.topK(1).isEmpty()) {
			throw new TranslateException("No prediction results available");
		}
		logPredictionResults(result);
	}

	private static void logPredictionResults(Classifications result) {
		List<Classifications.Classification> topK = result.topK(TOP_K_PREDICTIONS);

		logger.info("\nPrediction Results:");
		logger.info("=".repeat(40));

		for (int i = 0; i < topK.size(); i++) {
			Classifications.Classification c = topK.get(i);
			logger.info("{}. Digit {} - Confidence: {:.2f}%".replace("{:.2f}", String.format("%.2f", c.getProbability() * 100)), i + 1, c.getClassName());
		}

		logger.info("=".repeat(40) + "\n");
	}

	public static void main(String[] args) {
		if (args.length < 1) {
			logger.error("""
					
					Error: No image path provided
					Usage: java MnistPredictor <image-path>
					Example: java MnistPredictor src/main/resources/test-images/number_7.png
					""");
			System.exit(1);
		}

		try {
			Classifications result = predict(args[0]);
			List<Classifications.Classification> topK = result.topK(TOP_K_PREDICTIONS);

			logger.info("\nFinal Results:");
			logger.info("-".repeat(30));

			Classifications.Classification best = topK.get(0);
			logger.info("Most likely digit: {} with {}", best.getClassName(), String.format("%.2f%%", best.getProbability() * 100));

			logger.info("\nTop {} Predictions:", TOP_K_PREDICTIONS);
			for (int i = 0; i < topK.size(); i++) {
				Classifications.Classification c = topK.get(i);
				logger.info("{}. Digit {} - Confidence: {}", i + 1, c.getClassName(), String.format("%.2f%%", c.getProbability() * 100));
			}

			logger.info("-".repeat(30));
			logger.info("Prediction completed successfully\n");

		} catch (MalformedModelException e) {
			logger.error("\nModel Error: {}", e.getMessage());
			logger.error("Please ensure the model is properly trained");
			System.exit(1);
		} catch (IOException e) {
			logger.error("\nIO Error: {}", e.getMessage());
			logger.error("Please check if the image file exists and is accessible");
			System.exit(1);
		} catch (TranslateException e) {
			logger.error("\nTranslation Error: {}", e.getMessage());
			logger.error("Error processing the image. Please ensure it's a valid image file");
			System.exit(1);
		} catch (Exception e) {
			logger.error("\nUnexpected Error: {}", e.getMessage());
			logger.error("Please check the stack trace for more details");
			e.printStackTrace();
			System.exit(1);
		}
	}
}