# ColorPortal

ColorPortal是一个基于卷积神经网络的自编码器AI，用于自动化合成主要用于我的世界模组的像素材质贴图。  

## Why this?  

很多人可能会说，传统的工程方法，比如调整色相是不是完全可以达到类似的目标？  
是的，某些情况下确实可以如此，比如格雷科技模组就通过自动化色相调整来实现了大量的材质合成。  

但是，我们必须看见，这种方法具有很大的局限性，如果遇到非规则色相调整，譬如镐子，镐头和握把它们边缘和中心的
色相各不相同，如果只是单一的调整整个色相，会使得成品非常难看，大量贴图如果手动选择不同的色相区域分别调整，
也是一件累人的事情。  

通过AI调整就没有这么多烦恼，而且它可以自动学习色相区域、饱和度和阴影高光等特征并合成，更大程度地减少开发者负担。  

与此同时，“任何东西都是双刃剑”对于此项目所使用的方法也不例外，基于机器学习的方法不太可能达到人类手动通过
工程方法调整的色彩那么美观，也不太可能在少样本学习的情况下出令人满意的结果，最大的优势在于效率，抛开效率，
这个工具没有什么优势。  

为了快速开发原型并验证迭代，这个工具才得以产生，但它并不是用来替代模组艺术家的，任何使用此工具带来的不利后果，
由使用者自行承担。  

## 示例

[在B站观看演示](https://www.bilibili.com/video/BV1414y1M7t4/)

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

