package texture.echo;

import ai.djl.Device;
import ai.djl.translate.TranslateException;
import site.superice.modart.texture.data.EnrichmentDataSet;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;

public class EnrichTrainEcho {
    public static void main(String[] args) {
        var start = System.currentTimeMillis();
        var trainer = new TrainEcho();
        var trainList = new ArrayList<BufferedImage>(8);
        var testList = new ArrayList<BufferedImage>(8);
        {
            trainList.add(load("./DataSet/src/main/resources/color/ie-ingot/Aluminium Ingot.png"));
            trainList.add(load("./DataSet/src/main/resources/color/ie-ingot/Constantan Ingot.png"));
            trainList.add(load("./DataSet/src/main/resources/color/ie-ingot/Copper Ingot.png"));
            trainList.add(load("./DataSet/src/main/resources/color/ie-ingot/Electrum Ingot.png"));
            trainList.add(load("./DataSet/src/main/resources/color/ie-ingot/Lead Ingot.png"));
            trainList.add(load("./DataSet/src/main/resources/color/ie-ingot/Nickel Ingot.png"));
        }
        {
            testList.add(load("./DataSet/src/main/resources/color/ie-ingot/Silver Ingot.png"));
            testList.add(load("./DataSet/src/main/resources/color/ie-ingot/Steel Ingot.png"));
            testList.add(load("./DataSet/src/main/resources/color/ie-ingot/Uranium Ingot.png"));
        }
        var batchSize = 4;
        for (int i = 0; i < 4; i++) {
            System.out.println("Start training: tier " + i);
            try {
                trainer.train(Paths.get("./TextureArt/target/model/echo"), 8,
                        EnrichmentDataSet.builder().addImages(trainList).setSampling(batchSize, true)
                                .optDevice(Device.gpu()).build(),
                        EnrichmentDataSet.builder().addImages(testList).setSampling(batchSize, true)
                                .optDevice(Device.gpu()).build());
            } catch (TranslateException | IOException e) {
                e.printStackTrace();
            }
            System.out.println("End training: tier " + i);
        }
        System.out.println("Total time: " + (System.currentTimeMillis() - start) + "ms");
    }

    public static BufferedImage load(String path) {
        try {
            return ImageIO.read(new File(path));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
