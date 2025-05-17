from src.init import *
# from src.feature_engg.feat_engg import *  #uncomment this line for unit-testing 

class TreeModels:
	def __init__(self, combined_df=None, combined_df_t=None):
		self.combined_df = combined_df
		self.combined_df_t = combined_df_t

	def __verify_data(self):
		# other verifications can be added

		features_flag = set(self.combined_df.columns) - set(self.combined_df_t.columns)
		numeric_columns = self.combined_df_t.select_dtypes('number').columns
		# print(numeric_columns)
		null_sum_flag = self.combined_df[numeric_columns].isna().sum().sum() or self.combined_df_t[numeric_columns].isna().sum().sum()

		print(null_sum_flag)
		assert len(features_flag) == 1, "Feature mismatch, recheck columns"
		assert null_sum_flag == 0, f"Values missing, {null_sum_flag} unhandled null values"


	def __find_best_thres(self, y_val=None, y_preds=None):
		best_thres = 0.01
		best_roc = 0.0

		for thresh in np.arange(0.00, 1.0, 0.001):
			preds = (y_preds >= thresh).astype(int)
			roc = roc_auc_score(y_val, preds)
			if roc > best_roc:
				best_roc = roc
				best_thres = thresh

		return best_thres, best_roc

	def model(self):
		# model training and classification report
		self.__verify_data()

		final_data = self.combined_df.select_dtypes('number')
		X = final_data.drop(columns=['TARGET'])
		y = final_data['TARGET']
		sc = StandardScaler()
		X_scaled = sc.fit_transform(X)
		X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, stratify=y, test_size=0.15, random_state=42)
		xgbc = XGBClassifier(n_estimators=100, max_depth=4, min_child_weight=4, subsample=0.8, scale_pos_weight=0.35, eval_metric='auc', booster='gbtree')
		xgbc.fit(X_train, y_train)

		# cv_score = cross_val_score(xgbc, X_train, y_train, cv=kf, scoring='roc_auc', verbose=1)

		# y_preds = xgbc.predict_proba(X_val)
		# y_preds = np.array([0 if x>y else 1 for (x,y) in y_preds])

		y_preds = xgbc.predict_proba(X_val)[:, 1]

		best_thres, best_roc = self.__find_best_thres(y_val=y_val, y_preds=y_preds)
		print(f"Best threshold for ROC_AUC Score: {best_thres:.2f} → roc Score: {best_roc:.4f}")
		y_preds = (y_preds >= best_thres).astype(int)

		print(f"ROC_AUC Score for Basic XGB Model (model predictions) : { roc_auc_score(y_val, y_preds)}")
		print("F1 Score:", f1_score(y_val, y_preds))
		print("Confusion Matrix:\n", confusion_matrix(y_val, y_preds))



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
	feat_eng = FeatureEngineering(acc_data=acc_df, enq_data=enq_df, acc_data_t=acc_df_t, enq_data_t=enq_df_t, train_ops=train_ops, test_ops=test_ops)
	combined_df, combined_df_t = feat_eng.final_merged_dataset()


	obj = TreeModels(combined_df=combined_df, combined_df_t=combined_df_t)
	obj.model()

# Execute using from "loan-check" relative path - python -m src.model.model