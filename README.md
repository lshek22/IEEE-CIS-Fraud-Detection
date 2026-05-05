# IEEE-CIS Fraud Detection

## კონკურსის მიმოხილვა

ეს პროექტი შესრულებულია Kaggle-ის კონკურსის ფარგლებში — IEEE-CIS Fraud Detection. კონკურსის მიზანია საბანკო ტრანზაქციების თაღლითობის (fraud) გამოვლენა. მონაცემები მოდის IEEE Computational Intelligence Society-ისგან და შეიცავს რეალურ ტრანზაქციებს Vesta Corporation-ის სისტემიდან.
შეფასების მეტრიკა: ROC-AUC (Receiver Operating Characteristic — Area Under Curve)

## ჩემი მიდგომა
პრობლემა გადავჭერი ეტაპობრივად:

* **Cleaning** — NaN-ების დამუშავება, ზედმეტი feature-ების ამოშლა
* **Feature Engineering** — ახალი ინფორმაციული სვეტების შექმნა
* **Feature Selection** — კორელაციის ფილტრი ზედმეტი სვეტების ამოსაშლელად
* **Training** — სამი განსხვავებული არქიტექტურის გატესტვა: Logistic Regression, Random Forest, XGBoost
* **Inference** — საუკეთესო მოდელის (XGBoost) Model Registry-დან ჩატვირთვა და submission-ის გენერაცია

ყველა ექსპერიმენტი დალოგილია MLflow-ზე Dagshub-ის საშუალებით.

## რეპოზიტორიის სტრუქტურა
    House-Prices/
                ├── model_experiment_xgboost.ipynb           # XGBoost — cleaning, FE, FS, training
                ├── model_experiment_logistic_regression.ipynb  # Logistic Regression
                ├── model_experiment_random_forest.ipynb     # Random Forest
                ├── model_inference.ipynb    # საბოლოო პროგნოზი და submission
                └── README.md                # README

### ფაილების განმარტება:
model_experiment_*.ipynb — ყოველი მოდელისთვის ცალკე notebook, რომელიც მოიცავს Cleaning, Feature Engineering, Feature Selection და Training ეტაპებს, გამოყოფილია შესაბამისი Heading-ებით.
model_inference_ieee.ipynb — test set-ზე საბოლოო პროგნოზი. მოდელი ჩაიტვირთება Model Registry-დან (models:/IEEEFraud_BestModel/1) და მისი გამოძახებით გენერირდება submission.csv.

## Feature Engineering

ახალი Feature-ების შექმნა

| Feature | Description | Reason |
| :--- | :--- | :--- |
| **TransactionAmt_log** | `log1p(TransactionAmt)` | Distribution is heavy-tailed; log normalization helps model convergence. |
| **Transaction_hour** | `(TransactionDT / 3600) % 24` | Fraud patterns often correlate with specific times of day (e.g., late night). |
| **Transaction_day** | `(TransactionDT / 86400) % 7` | Weekly patterns can reveal behavioral anomalies. |
| **D1_normalized** | `TransactionDT / 86400 - D1` | D1 represents card age; normalization isolates the account's relative age. |
| **email_match** | `P_emaildomain == R_emaildomain` | Discrepancies between purchaser and recipient domains are fraud indicators. |
| **card1/2_freq, addr1_freq** | Frequency Encoding | Rare card or address combinations often signal fraudulent activity. |

კატეგორიული ცვლადების კოდირება
OrdinalEncoder — გამოვიყენე Pipeline-ში handle_unknown='use_encoded_value' (unknown=-1). კოდირება გამოყენებულია ყველა object-ტიპის სვეტზე.
NaN მნიშვნელობების დამუშავება
სამივე notebook-ში ერთიანი სტრატეგია გამოიყენეს:

* NULL > 50% — სვეტი მთლიანად წაიშალა (ძალიან ბევრი გამოტოვებული მნიშვნელობა, ინფორმაციული ღირებულება ნულთან ახლოს)
* რიცხვითი სვეტები — SimpleImputer(strategy='median') — მედიანა outlier-ების მიმართ მდგრადია
* კატეგორიული სვეტები — SimpleImputer(strategy='most_frequent') — ყველაზე გავრცელებული მნიშვნელობა

ეს სტრატეგიები ჩაშენებულია Pipeline-ში, ანუ ისინი test set-ზეც ავტომატურად გადის, train set-ის სტატისტიკის საფუძველზე — data leakage გამოირიცხა.

## Cleaning
NULL threshold-ის გამოყენება: ისეთი სვეტები, სადაც მნიშვნელობების 50%-ზე მეტი გამოტოვებულია, წაიშალა. ამ კრიტერიუმს დააკმაყოფილა მრავალი V-სვეტი (V1-V339 სერიებიდან) და identity ველები.


  ```python
  null_count = train.isnull().mean()
  cols_to_drop = [col for col in null_count[null_count > 0.5].index if col != 'isFraud']
  ```

      
ეს გავლენა MLflow-ზე დაილოგა: dropped_cols_count, original_col_count, final_col_count.

## Feature Selection
კორელაციის ფილტრი (threshold = 0.85)
მაღალ კორელირებული სვეტების წყვილებიდან ის სვეტი წაიშალა, რომელიც ნაკლებ ინფორმაციას ატარებდა. ეს XGBoost notebook-ში გამოყენებულ იქნა feature count-ის შემცირებისთვის.
```python
corr_matrix = train[num_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(...), k=1).astype(bool))
cols_to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
```
შედეგი: რამდენიმე ათეული კოლინეარული სვეტი ამოიშალა, მოდელი სწრაფად გაეშვა და overfitting-ი შემცირდა.
MLflow-ზე დაილოგა: corr_threshold, cols_dropped_count, final_feature_count.

## Training
ტესტირებული მოდელები
### 1. Logistic Regression (Random_Logistic_Regression ექსპერიმენტი)
Linear მოდელი გამოყენებულ იქნა baseline-ად. StandardScaler ჩართული იქნა Pipeline-ში, რადგან LR მგრძნობიარეა feature-ების სკალაზე.

| Run | C | val_AUC | შენიშვნა |
| :--- | :--- | :--- | :--- |
| **LR_C0001_underfit** | 0.0001 | ~0.84 | ძლიერი L2 — Underfitting |
| **LR_C1_balanced** | 1.0 | ~0.87 | `class_weight='balanced'` |

დასკვნა: Logistic Regression ამ არარეულ, მაღალი განზომილების მონაცემებზე საკმარისი სიძლიერის არ არის. val_AUC ~0.87 — მიუხედავად ამისა, კარგი baseline.


### 2. Random Forest (Random_Forest_Training ექსპერიმენტი)

მნიშვნელოვანი შენიშვნა: Random Forest-ს სრულ dataset-ზე გაშვება Kaggle Notebook-ზე ვერ მოხდა — ოპერატიული მეხსიერება ამოიწურა (MemoryError). ამ მიზეზით, Cross-Validation 50,000-ელემენტიანი sample-ით შეასრულა, ხოლო training — სრული train set-ით. ეს შეზღუდვა კომპრომისია — CV შედეგები შეიძლება ოდნავ განსხვავდებოდეს სრული მონაცემების შედეგებისგან.

სამი კონფიგურაცია გატესტდა — მიზნად underfitting-ისა და overfitting-ის ჩვენება:
| Run | n_estimators | max_depth | train_AUC | val_AUC | პრობლემა |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RF_Underfit_Depth3** | 50 | 3 | ~0.80 | ~0.78 | Underfitting — ძალიან მარტივი მოდელი |
| **RF_Balanced_Depth12** | 100 | 12 | ~0.92 | ~0.90 | საუკეთესო ბალანსი |
| **RF_Overfit_NoLimit** | 100 | `None` | ~0.99 | ~0.84 | Overfitting — train-ს ზეპირდება |

Underfitting vs Overfitting — Random Forest

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/2d98331b-276c-426d-b384-495ce5cf0873" />

      Underfitting ახსნა (Depth=3): ხე ძალიან მცირეა — ვერ სწავლობს მონაცემების სტრუქტურას. Train და Val AUC ორივე დაბალია (~0.80/0.78), gap მინიმალურია, მაგრამ        საერთო დონე სუსტია.

      Overfitting ახსნა (max_depth=None): შეუზღუდავი ხე train მონაცემს "ზეპირდება". Train AUC ~0.99 (თითქმის სრულყოფილი), val AUC ~0.84 — დიდი gap ნიშნავს,          რომ მოდელი ვერ განაზოგადებს ახალ მონაცემებზე.

      საუკეთესო (Depth=12): train/val gap მინიმალური, val AUC მაქსიმალური.

### 3. XGBoost (XGBoost_Training ექსპერიმენტი)
XGBoost — ყველაზე ძლიერი მოდელი ამ ტიპის სტრუქტურირებული მონაცემებისთვის. 6 განსხვავებული კონფიგურაცია გატესტდა:

| Run | params | train_AUC | val_AUC | cv_AUC | overfit_gap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGB_baseline_no_weight** | n=100, lr=0.1, depth=6 | ~0.97 | ~0.90 | ~0.89 | ~0.08 |
| **XGB_scale_pos_weight** | + class imbalance correction | ~0.97 | ~0.90 | ~0.89 | ~0.08 |
| **XGB_learning_rate_0.05_n200** | lr=0.05, n=200 | ~0.98 | ~0.91 | ~0.90 | ~0.08 |
| **XGB_depth3_underfit** | depth=3 | ~0.88 | ~0.86 | ~0.85 | ~0.03 |
| **XGB_subsample_colsample** | sub=0.8, col=0.8 | ~0.97 | ~0.91 | ~0.90 | ~0.07 |
| **XGB_min_child_weight5** | min_child=5, + sub/col | ~0.96 | ~0.91 | ~0.90 | ~0.06 |


**XGBoost Underfitting** ახსნა (Depth=3):


max_depth=3-ით XGBoost-ს ძალიან შეუზღუდული სივრცე აქვს. train_AUC მხოლოდ ~0.88, val_AUC ~0.86 — ორივე დაბალი. ეს კლასიკური underfitting-ია.


**scale_pos_weight**-ის მნიშვნელობა:
IEEE dataset-ში კლასები დაუბალანსებელია (~3.5% fraud). scale_pos_weight = n_negative / n_positive ≈ 27 — ამ პარამეტრის გამოყენება fraud კლასს მეტ წონას ანიჭებს training-ში, რაც recall-ს აუმჯობესებს.

**subsample** და **colsample_bytree**:
ყოველ ხეზე feature-ებისა და მაგალითების შემთხვევითი შერჩევა overfitting-ს ამცირებს — regularization-ის ეფექტური ფორმა.


**Hyperparameter** ოპტიმიზაცია
ყოველი გაშვებისას overfit_gap = train_auc - cv_auc_mean ილოგება. ეს მეტრიკა გვიჩვენებს, რამდენად "ზეპირდება" მოდელი training მონაცემს. მიზანი — gap-ის მინიმიზაცია val_AUC-ის მაქსიმიზაციის პირობებში.
საბოლოო მოდელის შერჩევა
საბოლოო მოდელი: XGBoost (subsample + colsample კონფიგურაცია)
შერჩევის მიზეზები:

* ყველაზე მაღალი val_AUC (~0.91) სამი არქიტექტურიდან
* overfit_gap შედარებით მცირეა (~0.07)
* cv_auc_mean სტაბილურია (~0.90), რაც ნიშნავს, რომ მოდელი კარგად განაზოგადებს სხვადასხვა split-ზე
* Pipeline-ად შენახვა უზრუნველყოფს test set-ზე პირდაპირ გაშვებას

Kaggle-ის საბოლოო შედეგი: 0.900180

Overfitting/Underfitting Bar Chart (XGBoost)
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/20c43c67-f1b8-4fbe-99d7-69c0c5295ece" />

Best feature for XGBoost
<img width="989" height="690" alt="image" src="https://github.com/user-attachments/assets/fb1b985e-ab14-408e-a630-43fcfad1e6bf" />


MLflow Tracking
ექსპერიმენტების ბმული
ყველა ექსპერიმენტი დალოგილია Dagshub-ზე:\


    https://dagshub.com/lshek22/IEEE-CIS-Fraud-Detection


ექსპერიმენტების სტრუქტურა


| ექსპერიმენტი | Run-ები |
| :--- | :--- |
| **XGBoost_Training** | `XGBoost_Cleaning`, `XGBoost_Feature_Engineering`, `XGBoost_Feature_Selection`, `XGB_Baseline_With_CV`, `XGB_baseline_no_weight`, `XGB_scale_pos_weight`, `XGB_learning_rate_0.05_n200`, `XGB_depth3_underfit`, `XGB_subsample_colsample`, `XGB_min_child_weight5`, `XGB_Best_Final` |
| **Random_Forest_Training** | `RF_Underfit_Depth3`, `RF_Balanced_Depth12`, `RF_Overfit_NoLimit` |
| **Logistic_Regression** | `LR_C0001_underfit`, `LR_C1_balanced`, `LR_Best_Final` |

ჩაწერილი მეტრიკები და პარამეტრები
Cleaning:

* null_threshold — NaN-ების drop-ის ზღვარი
* original_col_count, dropped_cols_count, final_col_count

Feature Engineering:

* engineered_features — შექმნილი feature-ების სია

Feature Selection:

* corr_threshold — კორელაციის ზღვარი
* cols_dropped_count, final_feature_count

Training (ყველა run):

* ``n_estimators``, ``max_depth``, ``learning_rate``, ``subsample``, ``colsample_bytree``, ``scale_pos_weight``, ``min_child_weight``
* ``train_auc``, ``val_auc``, ``cv_auc_mean``, ``overfit_gap``

Model Registry
საუკეთესო XGBoost pipeline დარეგისტრირებულია:
          
          models:/IEEEFraud_BestModel/1

inference notebook-ში ჩაიტვირთება პირდაპირ:
    ```python
    best_model = mlflow.sklearn.load_model("models:/IEEEFraud_BestModel/1")
    ```
### საუკეთესო მოდელის შედეგები
| მეტრიკა | მნიშვნელობა |
| :--- | :--- |
| **val_AUC (Kaggle notebook)** | ~0.91 |
| **cv_AUC_mean** | ~0.90 |
| **overfit_gap** | ~0.07 |
| **Kaggle Public Score** | 0.90018 |


