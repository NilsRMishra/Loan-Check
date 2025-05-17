from src.init import *
from src.data_processing.preprocess import *
from src.feature_engg.feat_engg import *
from src.model.model import *
from src.testing.test import *


if __name__ == "__main__":
	file_paths = [
		'data/train/train_flag.csv',
		'data/test/test_flag.csv',
		'data/train/accounts_data_train.json',
		'data/train/enquiry_data_train.json',
		'data/test/accounts_data_test.json',
		'data/test/enquiry_data_test.json',
	]
	ppp = Preprocess(in_path=file_paths, out_path="data/processed")
	# ppp.__open_files(file_paths[0])
	ppp.load_clean_files()

	files = os.listdir('data/processed')
	base_path = 'data/processed/'
	acc_df = pd.read_csv(base_path + files[0])
	acc_df_t = pd.read_csv(base_path + files[1])
	enq_df = pd.read_csv(base_path + files[2])
	enq_df_t = pd.read_csv(base_path + files[3])
	train_ops = pd.read_csv(base_path + files[5])
	test_ops = pd.read_csv(base_path + files[4])

	# print(files)
	feateng = FeatureEngineering(acc_data=acc_df, enq_data=enq_df, acc_data_t=acc_df_t, enq_data_t=enq_df_t, train_ops=train_ops, test_ops=test_ops)

	combined_df, combined_df_t = feateng.final_merged_dataset()

	mod = TreeModels(combined_df=combined_df, combined_df_t=combined_df_t)
	mod.model()