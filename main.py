from models import *
from pacs import *
from trainer import *
from utils import *

if __name__ == "__main__":
    fix_seed()
    pacs_handler = PACSHandler()
    train_dataset, test_dataset = pacs_handler.split_domain(test_domain='photo')
    # train_dataset, test_dataset = pacs_handler.split_domain(test_domain='art_painting')
    # train_dataset, test_dataset = pacs_handler.split_domain(test_domain='cartoon')
    # train_dataset, test_dataset = pacs_handler.split_domain(test_domain='sketch')

    res50_photo_normal = ResNet50(num_classes=7)
    trainer = Trainer(res50_photo_normal, train_dataset, test_dataset, 0.25, 128, 300, 0.001)
    trainer.train(swa_with_tau=False)   # True if SWA extended to tau axis
    trainer.test()
    trainer.print_acc()