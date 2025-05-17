import train_baseline
import train_structure_and_feature
import train_structure_and_feature_2
import train_structure_and_feature_3
import read_and_show_metrics

def main():
    train_baseline.main()
    train_structure_and_feature.main()
    train_structure_and_feature_2.main()
    train_structure_and_feature_3.main()

    read_and_show_metrics.main_auc_roc()
    read_and_show_metrics.main_loss()

if __name__ == "__main__":
    main()