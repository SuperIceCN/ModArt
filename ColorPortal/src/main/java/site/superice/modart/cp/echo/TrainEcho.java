package texture.echo;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.metric.Metric;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.*;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.translate.TranslateException;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import site.superice.modart.texture.data.EchoDataSet;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class TrainEcho {
    public static int DEFAULT_BATCH_SIZE = 64;
    @NotNull
    protected Logger log;
    @NotNull
    public final Map<String, double[]> evaluatorMetrics = new HashMap<>();

    public TrainEcho() {
        this(null);
    }

    public TrainEcho(@Nullable Logger log) {
        if (log == null) {
            this.log = LoggerFactory.getLogger(TrainEcho.class);
        } else {
            this.log = log;
        }
    }

    public static void main(String[] args) {
        var start = System.currentTimeMillis();
        var trainer = new TrainEcho();
        // 480
        for (int i = 0; i < 16 ; i++) {
            System.out.println("Start training: tier " + i);
            try {
                trainer.train(Paths.get("./TextureArt/target/model/echo"), 4,
                        EchoDataSet.builder().length(512 * 32).setSampling(DEFAULT_BATCH_SIZE, true)
                                .optDevice(Device.gpu()).build(),
                        EchoDataSet.builder().length(512).setSampling(DEFAULT_BATCH_SIZE, true)
                                .optDevice(Device.gpu()).build());
            } catch (TranslateException | IOException e) {
                e.printStackTrace();
            }
            System.out.println("End training: tier " + i);
        }
        System.out.println("Total time: " + (System.currentTimeMillis() - start) + "ms");
    }

    public void train(@NotNull Path savePath, int numEpochs, @NotNull RandomAccessDataset trainDataSet,
                             @NotNull RandomAccessDataset testDataSet) throws TranslateException, IOException {
        var start = System.currentTimeMillis();

        trainDataSet.prepare();
        testDataSet.prepare();

        var optimizer = Optimizer.adam().build();
        var loss = Loss.l2Loss();

        DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                .optOptimizer(optimizer)
                .optDevices(new Device[]{Device.gpu()})
                .addEvaluator(loss)
                .addTrainingListeners(TrainingListener.Defaults.logging());

        try (Model model = Model.newInstance("mlp")) {
            model.setBlock(loadModelBlocks());
            // 继续训练
            if (savePath.toFile().exists()) {
                try {
                    model.load(savePath);
                } catch (MalformedModelException ignored) {
                    log.warn("Model is malformed, start training from scratch.");
                }
            }

            var preparedTime = System.currentTimeMillis();

            try (Trainer trainer = model.newTrainer(config)) {

                trainer.initialize(new Shape(3, 32, 32));
                trainer.setMetrics(new Metrics());

                EasyTrain.fit(trainer, numEpochs, trainDataSet, testDataSet);
                var metrics = trainer.getMetrics();

                trainer.getEvaluators()
                        .forEach(evaluator -> {
                            evaluatorMetrics.put("train_epoch_" + evaluator.getName(), metrics.getMetric("train_epoch_" + evaluator.getName()).stream()
                                    .mapToDouble(Metric::getValue).toArray());
                            evaluatorMetrics.put("validate_epoch_" + evaluator.getName(), metrics.getMetric("validate_epoch_" + evaluator.getName()).stream()
                                    .mapToDouble(Metric::getValue).toArray());
                        });
            }

            log.info("Preparing time: " + (preparedTime - start) + "ms");
            log.info("Training time: " + (System.currentTimeMillis() - preparedTime) + "ms");

            if (!savePath.toFile().exists()) {
                //noinspection ResultOfMethodCallIgnored
                savePath.toFile().mkdirs();
            }
            var result = savePath.toFile().listFiles();
            if (result != null && result.length > 0) {
                //noinspection ResultOfMethodCallIgnored
                Objects.requireNonNull(savePath.toFile().listFiles())[0].delete();
            }
            model.save(savePath, "mlp");
        }
    }

    @NotNull
    protected Block loadModelBlocks() {
        return buildBlocks();
    }

    @NotNull
    public static Block buildBlocks() {
        var net = new SequentialBlock();
        net.add(new LambdaBlock(input -> {
                    var i = input.get(0);
//                    System.out.println("colorNetInputShape: " + i.getShape());
                    if (i.getShape().dimension() == 3) {
                        var tmp = i.reshape(new Shape(1, 3, 32, 32));
//                        System.out.println("colorNetOutputShape: " + tmp.getShape());
                        return new NDList(tmp);
                    } else {
//                        System.out.println("colorNetOutputShape: " + i.getShape());
                        return new NDList(i);
                    }
                }))
                .add(Conv2d.builder().setKernelShape(new Shape(9, 9))
                        .optStride(new Shape(1, 1))
                        .setFilters(32 * 3)
                        .build())
                .add(Activation.leakyReluBlock(0.2f))
                .add(Pool.maxPool2dBlock(new Shape(5, 5), new Shape(1, 1), new Shape(2, 2)))
                .add(Conv2d.builder().setKernelShape(new Shape(9, 9))
                        .optStride(new Shape(1, 1))
                        .setFilters(16 * 3)
                        .build())
                .add(Activation.leakyReluBlock(0.2f))
                .add(Pool.maxPool2dBlock(new Shape(5, 5), new Shape(1, 1), new Shape(2, 2)))
                .add(Conv2d.builder().setKernelShape(new Shape(3, 3))
                        .optStride(new Shape(1, 1))
                        .optPadding(new Shape(1, 1))
                        .setFilters(8 * 3)
                        .build())
                .add(Activation.leakyReluBlock(0.2f))
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(12 * 12 * 3).optBias(true).build())
                .add(Activation.leakyReluBlock(0.05f))
                .add(Linear.builder().setUnits(12 * 12 * 3).optBias(true).build())
                .add(Activation.leakyReluBlock(0.05f))
                .add(Linear.builder().setUnits(16 * 16 * 3).build())
                .add(Activation::sigmoid)
                .add(Blocks.batchFlattenBlock())
                .add(new LambdaBlock(input -> {
                    var i = input.singletonOrThrow();
                    var shape = i.getShape();
//                    System.out.println("FinalInputShape: " + shape);
                    var tmp = i.reshape(new Shape(shape.get(0), 3, 16, 16));
//                    System.out.println("FinalOutputShape: " + tmp.getShape());
                    return new NDList(tmp);
                }));
                net.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
                net.setInitializer(new NormalInitializer(), Parameter.Type.BIAS);
        return net;
    }
}
