from torch import nn


# built as https://github.com/ricvolpi/generalize-unseen-domains/blob/master/model.py
class MnistModel(nn.Module):
    def __init__(self, n_classes=100):
        super().__init__()
        
        outfeats = 1024 
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 5 * 5, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
        )

        print("Using LeNet (%d)" % outfeats)
        self.class_classifier = nn.Linear(outfeats, n_classes)

    def get_params(self, base_lr):
        raise "No pretrained exists for LeNet - use train all"

    def forward(self, x, lambda_val=0):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        last_hidden_features = x
        return last_hidden_features, self.class_classifier(x)


def lenet(classes):
    model = MnistModel(classes)
    return model
