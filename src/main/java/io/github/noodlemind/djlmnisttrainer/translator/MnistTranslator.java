package io.github.noodlemind.djlmnisttrainer.translator;

import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MnistTranslator implements Translator<Image, Classifications> {
	private static final Logger logger = LoggerFactory.getLogger(MnistTranslator.class);
	private final List<String> classes;

	public MnistTranslator() {
		this.classes = IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
	}

	@Override
	public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
		NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);

		array = NDImageUtils.resize(array, Mnist.IMAGE_HEIGHT, Mnist.IMAGE_WIDTH);
		logger.debug("Resized to: {}x{}", Mnist.IMAGE_WIDTH, Mnist.IMAGE_HEIGHT);

		array = array.div(255.0f);

		if (isWhiteBackground(array)) {
			array = ctx.getNDManager().create(1.0f).sub(array);
		}

		array = array.reshape(Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH);
		logger.debug("Final shape: {}", array.getShape());

		return new NDList(array);
	}


	@Override
	public Classifications processOutput(TranslatorContext ctx, NDList list) throws Exception {
		NDArray probabilities = list.singletonOrThrow();
		probabilities = probabilities.softmax(0);
		logger.debug("Output probabilities shape: {}", probabilities.getShape());
		return new Classifications(classes, probabilities);
	}

	@Override
	public Batchifier getBatchifier() {
		return Batchifier.STACK;
	}

	private boolean isWhiteBackground(NDArray array) {
		float[] data = array.toFloatArray();
		float sum = 0;
		int borderPixels = 0;

		for (int i = 0; i < Mnist.IMAGE_WIDTH; i++) {
			sum += data[i];
			sum += data[(Mnist.IMAGE_HEIGHT - 1) * Mnist.IMAGE_WIDTH + i];
			borderPixels += 2;
		}

		for (int i = 1; i < Mnist.IMAGE_HEIGHT - 1; i++) {
			sum += data[i * Mnist.IMAGE_WIDTH];
			sum += data[i * Mnist.IMAGE_WIDTH + Mnist.IMAGE_WIDTH - 1];
			borderPixels += 2;
		}

		float avgBorderValue = sum / borderPixels;
		logger.debug("Average border value: {}", avgBorderValue);
		return avgBorderValue > 0.5f;
	}
}