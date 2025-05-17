from src.init import *
# from init import *

class Preprocess:
	def __init__(self, in_path={}, out_path=""):
		self.in_path = in_path
		self.out_path = out_path
		self.today_date = pd.Timestamp.today().date()

	def __open_files(self, path=""):
		try:
			if(path.endswith(".csv")):
				print(f"Loading {path.split('/')[-1]} ...")
				data = pd.read_csv(path)
				return data
			
			elif(path.endswith(".json")):
				print(f"Loading {path.split('/')[-1]} ...")
				with open(path, 'r') as f:
					data = json.loads(f.read())

				data = [i for items in data for i in items]
				data = pd.DataFrame(data)
				return data
			else:
				raise Exception("File Path Invalid")
		except:
			print("Invalid File Path")
		
	def __payment_string_split(self, x):
		return [int(x[i:i+3]) for i in range(0, len(x), 3)]

	def __close_date_filler(self, item):
		if(pd.isna(item['closed_date'])):
			delta = len(item['payment_hist_string'])
			current_date = pd.to_datetime(item['open_date'])
			overdue_flag = item['amount_overdue']!=0
			if(overdue_flag):
				item['closed_date'] = str(self.today_date)
			else:
				item['closed_date'] = str((current_date + pd.DateOffset(months=delta)).date())
		return item['closed_date']


	def load_clean_files(self):
		train_ops, test_ops, acc_data, enq_data, acc_data_t, enq_data_t = map(self.__open_files, self.in_path)
		# self.train_ops, self.test_ops, self.acc_data, self.enq_data, self.acc_data_t, self.enq_data_t = map(self.__open_files, self.in_path)
		# print(train_ops.head(), type(train_ops))

		# Accounts Data Cleanup

		acc_data['payment_hist_string'] = acc_data['payment_hist_string'].apply(self.__payment_string_split)
		acc_data_t['payment_hist_string'] = acc_data_t['payment_hist_string'].apply(self.__payment_string_split)

		acc_data['closed_date'] = acc_data[['closed_date', 'open_date', 'payment_hist_string', 'amount_overdue']].apply(self.__close_date_filler, axis=1)
		acc_data_t['closed_date'] = acc_data_t[['closed_date', 'open_date', 'payment_hist_string', 'amount_overdue']].apply(self.__close_date_filler, axis=1)

		# acc_data = acc_data.dropna(axis=0)
		# acc_data_t = acc_data_t.dropna(axis=0)

		# Enquiry Data Cleanup
		# ---

		acc_data.to_csv(f'{self.out_path}/acc_data_cleaned.csv', index=False)
		acc_data_t.to_csv(f'{self.out_path}/acc_data_cleaned_t.csv', index=False)

		enq_data.to_csv(f'{self.out_path}/enquiry_data_cleaned.csv', index=False)
		enq_data_t.to_csv(f'{self.out_path}/enquiry_data_cleaned_t.csv', index=False)

		train_ops.to_csv(f'{self.out_path}/train_data.csv', index=False)
		test_ops.to_csv(f'{self.out_path}/test_data.csv', index=False)

		
	
		

		




if __name__ == "__main__":
	file_paths = [
		'data/train/train_flag.csv',
		'data/test/test_flag.csv',
		'data/train/accounts_data_train.json',
		'data/train/enquiry_data_train.json',
		'data/test/accounts_data_test.json',
		'data/test/enquiry_data_test.json',
	]
	x = Preprocess(in_path=file_paths, out_path="data/processed")
	# x.__open_files(file_paths[0])
	x.load_clean_files()

	# print("Reserved for Unit Testing. Execute main.py.")



# Execute using from "loan-check" relative path - python -m src.data_preprocessing.preprocess