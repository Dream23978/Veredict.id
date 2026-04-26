Parameter tuning:
model = XGBClassifier(
    n_estimators=40,
    learning_rate=0.05,
    max_depth=1,              
    min_child_weight=15,      
    gamma=0.8,                 
    subsample=0.5,            
    colsample_bytree=0.8,     
    reg_alpha=1.0,            
    reg_lambda=1.0,           
)
