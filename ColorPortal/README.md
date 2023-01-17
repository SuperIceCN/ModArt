# ColorPortal

ColorPortal是一个基于卷积神经网络的自编码器AI，用于自动化合成主要用于我的世界模组的像素材质贴图。  

## 示例

等待编写

## 使用

### 从DataSet子项目中构建训练用数据集

```java
import site.superice.modart.cp.data.EchoDataSet;

public class BuildFromDataSetSubProject {
    public static void fun() {
        var dataset = EchoDataSet.builder().length(aNumber * 32).setSampling(batchSize, true)
                .optDevice(Device.gpu()) // If you're not using GPUs, just delete this line
                .build();
    }
}
```

### 自定义迁移学习用数据集

```java
import site.superice.modart.cp.data.EnrichmentDataSet;

import java.util.ArrayList;

public class BuildEnrichmentDataSet {
    public static void fun() {
        var trainList = new ArrayList<BufferedImage>();
        {
            trainList.add(bufferdImage1);
            trainList.add(bufferdImage2);
            // ......
        }
        var dataset = EnrichmentDataSet.builder().addImages(trainList).setSampling(batchSize, true)
                .optDevice(Device.gpu()) // If you're not using GPUs, just delete this line
                .build();
    }
}
```

### 训练神经网络

```java
import site.superice.modart.cp.echo.TrainEcho;

import java.nio.file.Paths;

public class Train {
    public static void fun() {
        var train = new TrainEcho();
        trainer.train(Paths.get("path/to/dir/of/the/model"), numEpochs, trainDataSet, validDataSet);
    }
}
```

### 合成新图片

```java
import ai.djl.Model;
import ai.djl.translate.TranslateException;
import site.superice.modart.cp.echo.RepaintEcho;
import site.superice.modart.cp.echo.TrainEcho;

import java.nio.file.Paths;

public class ProcessNewImage {
    public static void fun() throws TranslateException {
        try (var model = Model.newInstance("cp")) { // cp is the name of the base model
            model.setBlock(TrainEcho.buildBlocks());
            model.load(Paths.get("path/to/dir/of/the/model"));
            var repaintedImage = RepaintEcho.repaint(model, shapeImage, colorImage);
        }
    }
}
```

## 注意事项  

- 首次运行需要连接互联网，程序将会自动下载并配置PyTorch环境
- 如需使用CUDA加速，请提前预装CUDA11+及匹配版本的CUDNN

