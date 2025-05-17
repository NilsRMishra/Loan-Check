from src.init import *


class FeatureEngineering:
	def __init__(self, train_ops=None, test_ops=None, acc_data=None, enq_data=None, acc_data_t=None, enq_data_t=None):
		self.train_ops, self.test_ops, self.acc_data, self.enq_data, self.acc_data_t, self.enq_data_t = \
		train_ops, test_ops, acc_data, enq_data, acc_data_t, enq_data_t

		self._user_dummies = None
		self._enq_dummies = None
		
	def __date_based_features(self, df=None, is_acc=True, printing=None):
		# date based feature extraction
		
		print(f"Working on {printing} ...")

		def column_date_based_features(c_df=None, column='date'):

			pre = column[0]
			c_df[f'{pre}_date'] = c_df[column].dt.day
			c_df[f'{pre}_day'] = c_df[column].dt.day_of_week+1
			c_df[f'{pre}_month'] = c_df[column].dt.month
			c_df[f'{pre}_year'] = c_df[column].dt.year
			c_df[f'{pre}_quarter'] = c_df[column].dt.quarter
			c_df[f'{pre}_is_weekday'] = c_df[column].dt.dayofweek < 5

			return c_df

		if(is_acc):
			df['open_date'] = pd.to_datetime(df['open_date'])
			df['closed_date'] = pd.to_datetime(df['closed_date'])
			df = column_date_based_features(c_df=df, column='open_date')
			df = column_date_based_features(c_df=df, column='closed_date')
			df['loan_duration_m'] = round((df['closed_date'] - df['open_date']) / pd.Timedelta(30.41, 'D'))
		else:
			df['enquiry_date'] = pd.to_datetime(df['enquiry_date'])
			df = column_date_based_features(c_df=df, column='enquiry_date')			
		
		# __date_based_features(acc_data.iloc[:10], column='open_date')
		return df

	def __user_features(self, df=None, printing=None):
		# additional flag based feature extraction

		print(f"Working on {printing} ...")

		df['payment_hist_string'] = df['payment_hist_string'].apply(lambda x: eval(x))
		df['max_delay'] = df['payment_hist_string'].apply(lambda x: max(x) if len(x)>0 else 0)
		df['avg_delay'] = df['payment_hist_string'].apply(lambda x: np.mean(x) if len(x)>0 else 0)
		df['is_loan_active'] = df['amount_overdue']>0
		df[df.select_dtypes(bool).columns] = df.select_dtypes(bool).astype(int)
		user_dummies = pd.get_dummies(df['credit_type'], drop_first=True, prefix="c", dtype=int)
		self._user_dummies = user_dummies.columns
		df = pd.concat([df, user_dummies], axis=1)
		return df

	def __enquiry_features(self, df=None, printing=None):
		# binarized features for categorical columns
		print(f"Working on {printing} ...")
		enq_dummies = pd.get_dummies(df['enquiry_type'], drop_first=True, prefix="e", dtype=int)
		df = pd.concat([df, enq_dummies], axis=1)
		self._enq_dummies = enq_dummies.columns
		# print(self._enq_dummies)
		return df

	def __numerical_features(self, df=None, is_acc=True, printing=None):
		# aggregations for numerical features
		if(is_acc):
			print(f"Working on {printing} ...")

			agg_funcs = {
				'loan_amount': ['count', 'sum', 'mean', 'max', 'min'],
				'amount_overdue': ['sum', 'mean', 'max', 'min'],
				'open_date': ['max', 'min'],
				'closed_date': ['max', 'min'],
				'max_delay': ['sum', 'max'],
				'avg_delay': ['sum', 'max', 'min'],
				'loan_duration_m': ['mean', 'max', 'min'],
				'is_loan_active': ['max'],
				'credit_type': pd.Series.nunique
			}
			agg_numerics = df.groupby('uid').agg(agg_funcs).reset_index()
			agg_other = df.groupby('uid')[['o_date', 'o_day', 'o_month', 'o_year', 'o_quarter', 'o_is_weekday', 'c_date', \
										'c_day', 'c_month', 'c_year', 'c_quarter', 'c_is_weekday']].agg(['first', 'last']).reset_index()
			agg_binary = df.groupby('uid')[self._user_dummies].agg(['max']).reset_index()

		else:
			print(f"Working on {printing} ...")

			agg_funcs = {
				'enquiry_amt': ['count', 'sum', 'mean', 'max', 'min'],
				'enquiry_date':['max', 'min'],
				'enquiry_type': pd.Series.nunique
			}

			agg_numerics = df.groupby('uid').agg(agg_funcs).reset_index()
			agg_other = df.groupby('uid')[['e_date', 'e_day', 'e_month', 'e_year', 'e_quarter', 'e_is_weekday']].agg(['first', 'last']).reset_index()
			agg_binary = df.groupby('uid')[self._enq_dummies].agg(['max']).reset_index()

		agg_numerics.columns = ['_'.join(col).strip() for col in agg_numerics.columns]
		agg_other.columns = ['_'.join(col).strip() for col in agg_other.columns]
		agg_binary.columns = ['_'.join(col).strip() for col in agg_binary.columns]

		df = agg_numerics.merge(agg_other, on='uid_', how='outer').merge(agg_binary, on='uid_', how='outer')

		return df


	def __final_cleanup(self, df=None, printing=None):
		print(f"Working on {printing} ...")
		# filling null values in binary columns
		binary_columns = df.columns[df.nunique() == 2]
		df[binary_columns[:-1]] = df[binary_columns[:-1]].fillna(0)
		# filling null values in numeric_columns
		numeric_columns = df.select_dtypes('number').columns
		df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

		return df


	def final_merged_dataset(self):
		# final aggregation of dataset and cleanup
		self.acc_data = self.__date_based_features(df=self.acc_data, printing="acc_data_date_based_features")
		self.acc_data_t = self.__date_based_features(df=self.acc_data_t, printing="acc_data_t_date_based_features")
		
		self.acc_data = self.__user_features(df=self.acc_data, printing="acc_data_user_features")
		self.acc_data = self.__numerical_features(df=self.acc_data, is_acc=True, printing="acc_data_numerical_features")

		self.acc_data_t = self.__user_features(df=self.acc_data_t, printing="acc_data_t_user_features")
		self.acc_data_t = self.__numerical_features(df=self.acc_data_t, is_acc=True, printing="acc_data_t_numerical_features")

		self.enq_data = self.__date_based_features(df=self.enq_data, is_acc=False, printing="enq_data_date_based_features")
		self.enq_data_t = self.__date_based_features(df=self.enq_data_t, is_acc=False, printing="enq_data_t_date_based_features")

		self.enq_data = self.__enquiry_features(self.enq_data, printing="enq_data_enquiry_features")
		# print(self.enq_data.columns)
		self.enq_data = self.__numerical_features(self.enq_data, is_acc=False, printing="enq_data_numerical_features")

		self.enq_data_t = self.__enquiry_features(self.enq_data_t, printing="enq_data_t_enquiry_features")
		self.enq_data_t = self.__numerical_features(self.enq_data_t, is_acc=False, printing="enq_data_t_numerical_features")

		self.acc_data_t[['c_Interbank credit_max', 'c_Loan for purchase of shares (margin lending)_max', 'c_Mobile operator loan_max']] = 0

		print(set(self.acc_data.columns) == set(self.acc_data_t.columns))
		# print(set(self.acc_data.columns) - set(self.acc_data_t.columns), set(self.acc_data_t.columns) - set(self.acc_data.columns))


		combined_df = self.acc_data.merge(self.enq_data, on='uid_', how='outer').merge(self.train_ops, left_on='uid_', right_on='uid', how='outer')
		combined_df_t = self.acc_data_t.merge(self.enq_data_t, on='uid_', how='outer').merge(self.test_ops, left_on='uid_', right_on='uid', how='outer')
		
		combined_df = self.__final_cleanup(combined_df, printing="combined_df_final_cleanup")
		combined_df_t = self.__final_cleanup(combined_df_t, printing="combined_df_t_final_cleanup")


		# print(self.acc_data.columns, self.acc_data_t.columns, self.enq_data.columns, self.enq_data_t.columns)
		# print(combined_df.shape, combined_df_t.shape)
		# print(combined_df.isna().sum())

		return combined_df, combined_df_t

if __name__ == "__main__":
	files = os.listdir('data/processed')
	base_path = 'data/processed/'
	acc_df = pd.read_csv(base_path + files[0])
	acc_df_t = pd.read_csv(base_path + files[1])
	enq_df = pd.read_csv(base_path + files[2])
	enq_df_t = pd.read_csv(base_path + files[3])
	train_ops = pd.read_csv(base_path + files[5])
	test_ops = pd.read_csv(base_path + files[4])

	# print(files)
	obj = FeatureEngineering(acc_data=acc_df, enq_data=enq_df, acc_data_t=acc_df_t, enq_data_t=enq_df_t, train_ops=train_ops, test_ops=test_ops)
	obj.final_merged_dataset()




# Execute using from "loan-check" relative path - python -m src.feature_engg.feat_engg