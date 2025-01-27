package io.github.noodlemind.djlmnisttrainer;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDList;
import ai.djl.translate.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public final class MnistPredictor {
	private static final Logger logger = LoggerFactory.getLogger(MnistPredictor.class);
	private static final Path MODEL_DIR = Paths.get("build/model");
	private static final String MODEL_NAME = "mnist";
	private static final String PARAMS_FILE = MODEL_NAME + "-0002.params";

	public static Classifications predict(String imagePath)
			throws IOException, TranslateException, MalformedModelException {
		Path imageFile = Paths.get(imagePath);
		logger.info("Loading image: {}", imageFile);

		// Verify model file exists
		Path paramsPath = MODEL_DIR.resolve(PARAMS_FILE);
		if (!Files.exists(paramsPath)) {
			throw new IOException("Model parameters not found at: " + paramsPath);
		}
		logger.info("Found model parameters at: {}", paramsPath);

		try (Model model = Model.newInstance(MODEL_NAME)) {
			model.setBlock(new Mlp(
					Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH,
					Mnist.NUM_CLASSES,
					new int[]{128, 64}));

			logger.info("Loading model from: {}", MODEL_DIR);
			model.load(MODEL_DIR, MODEL_NAME);

			Translator<Image, Classifications> translator = new Translator<>() {
				@Override
				public NDList processInput(TranslatorContext ctx, Image input)
						throws TranslateException {
					try {
						NDList list = new NDList();
						list.add(input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE));
						return list;
					} catch (Exception e) {
						throw new TranslateException("Failed to process input", e);
					}
				}

				@Override
				public Classifications processOutput(TranslatorContext ctx, NDList list)
						throws TranslateException {
					try {
						List<String> classes = IntStream.range(0, 10)
								                       .mapToObj(String::valueOf)
								                       .collect(Collectors.toList());
						return new Classifications(classes, list.singletonOrThrow());
					} catch (Exception e) {
						throw new TranslateException("Failed to process output", e);
					}
				}

				@Override
				public Batchifier getBatchifier() {
					return Batchifier.STACK;
				}
			};

			try (Predictor<Image, Classifications> predictor = model.newPredictor(translator)) {
				Image img = ImageFactory.getInstance().fromFile(imageFile);
				Classifications result = predictor.predict(img);
				logger.info("Prediction result: {}", result);
				return result;
			}
		}
	}

	public static void main(String[] args) {
		if (args.length < 1) {
			logger.error("Usage: java MnistPredictor <image-path>");
			System.exit(1);
		}

		try {
			Classifications result = predict(args[0]);
			Classifications.Classification best = result.best();
			logger.info("Best class: {} (probability: {:.2f}%)",
					best.getClassName(),
					best.getProbability() * 100);
		} catch (MalformedModelException e) {
			logger.error("Model format error: {}", e.getMessage(), e);
			System.exit(1);
		} catch (IOException e) {
			logger.error("I/O error: {}", e.getMessage(), e);
			System.exit(1);
		} catch (TranslateException e) {
			logger.error("Translation error: {}", e.getMessage(), e);
			System.exit(1);
		}
	}
}