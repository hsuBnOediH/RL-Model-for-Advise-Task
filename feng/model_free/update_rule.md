# Updating Rule and Forgeting Rule for RL model on Advise task
## (start,left)
### 1. Conneted Version
#### 1.1. Update
- (start,left) = (start,left) + lr * (actual_reward - (start,left))
- (start,right) = (start,right) + lr * (opposite_reward - (start,right))
#### 1.2. Forget  
- (start,advise) = (start,advise) + fr * (start_advise_intit_vale - (start,advise))
- (advise_left,left) = (advise_left,left) + fr * (advise_left_left_intit_vale - (advise_left,left))
- (advise_left,right) = (advise_left,right) + fr * (advise_left_right_intit_vale - (advise_left,right))
- (advise_right,left) = (advise_right,left) + fr * (advise_right_left_intit_vale - (advise_right,left))
- (advise_right,right) = (advise_right,right) + fr * (advise_right_right_intit_vale - (advise_right,right))
### 2. Disconneted Version
#### 2.1. Update
- (start,left) = (start,left) + lr * (actual_reward - (start,left))
#### 2.2. Forget
- (start,right) = (start,right) + fr * (start_right_intit_vale - (start,right))
- (start,advise) = (start,advise) + fr * (start_advise_intit_vale - (start,advise))
- (advise_left,left) = (advise_left,left) + fr * (advise_left_left_intit_vale - (advise_left,left))
- (advise_left,right) = (advise_left,right) + fr * (advise_left_right_intit_vale - (advise_left,right))
- (advise_right,left) = (advise_right,left) + fr * (advise_right_left_intit_vale - (advise_right,left))
- (advise_right,right) = (advise_right,right) + fr * (advise_right_right_intit_vale - (advise_right,right))

## (start,right)
### 1. Conneted Version
#### 1.1. Update
- (start,left) = (start,left) + lr * (opposite_reward - (start,left))
- (start,right) = (start,right) + lr * (actual_reward - (start,right))
#### 1.2. Forget
- (start,advise) = (start,advise) + f r (start_advise_intit_vale - (start,advise))
- (advise_left,left) = (advise_left,left) + fr * (advise_left_left_intit_vale - (advise_left,left))
- (advise_left,right) = (advise_left,right) + fr * (advise_left_right_intit_vale - (advise_left,right))
- (advise_right,left) = (advise_right,left) + fr * (advise_right_left_intit_vale - (advise_right,left))
- (advise_right,right) = (advise_right,right) + fr * (advise_right_right_intit_vale - (advise_right,right))

### 2. Disconneted Version
#### 2.1. Update
- (start,right) = (start,right) + lr * (actual_reward - (start,right))
#### 2.2. Forget
- (start,left) = (start,left) + fr * (start_left_intit_vale - (start,left))
- (start,advise) = (start,advise) + fr * (start_advise_intit_vale - (start,advise))
- (advise_left,left) = (advise_left,left) + fr * (advise_left_left_intit_vale - (advise_left,left))
- (advise_left,right) = (advise_left,right) + fr * (advise_left_right_intit_vale - (advise_left,right))
- (advise_right,left) = (advise_right,left) + fr * (advise_right_left_intit_vale - (advise_right,left))
- (advise_right,right) = (advise_right,right) + fr * (advise_right_right_intit_vale - (advise_right,right))

## (start,advise)
### 1. Conneted Version
#### 1.1. Update
- (start,advise) = (start,advise) + lr * (max_furture_reward - (start,advise))
#### 1.2. Forget
- forgeting rule is not applied to (start,advise)

### 2. Disconneted Version
#### 2.1. Update
- (start,advise) = (start,advise) + lr * (max_furture_reward - (start,advise))
#### 2.2. Forget
- forgeting rule is not applied to (start,advise)



## (advise_left,left)
### 1. Conneted Version
#### 1.1. Update
- (advise_left,left) = (advise_left,left) + lr * (actual_reward - (advise_left,left))
- (advise_left,right) = (advise_left,right) + lr * (opposite_reward - (advise_left,right))
- (advise_right,left) = (advise_right,left) + lr * (opposite_reward - (advise_right,left))
- (advise_right,right) = (advise_right,right) + lr * (actual_reward - (advise_right,right))

- (start,left) = (start,left) + without_advise_lr * (without_advise_actual_reward - (start,left))
- (start,right) = (start,right) + without_advise_lr * (without_advise_opposite_reward - (start,right))

#### 1.2. Forget
- No forgeting rule is applied to (advise_left,left)

### 2. Disconneted Version
#### 2.1. Update
- (advise_left,left) = (advise_left,left) + lr * (actual_reward - (advise_left,left))
- (advise_right,right) = (advise_right,right) + lr * (actual_reward - (advise_right,right))
- (start,left) = (start,left) + without_advise_lr * (without_advise_actual_reward - (start,left))
#### 2.2. Forget
- (advise_left,right) = (advise_left,right) + fr * (advise_left_right_intit_vale - (advise_left,right))
- (advise_right,left) = (advise_right,left) + fr * (advise_right_left_intit_vale - (advise_right,left))
- (start,right) = (start,right) + without_advise_fr * (start_right_intit_vale - (start,right))

## (advise_left,right)
### 1. Conneted Version
#### 1.1. Update
- (advise_left,right) = (advise_left,right) + lr * (actual_reward - (advise_left,right))
- (advise_right,left) = (advise_right,left) + lr * (actual_reward - (advise_right,left))
- (advise_right,right) = (advise_right,right) + lr * (opposite_reward - (advise_right,right))                
- (advise_left,left) = (advise_left,left) + lr * (opposite_reward - (advise_left,left))
- (start,left) = (start,left) + without_advise_lr * (without_advise_actual_reward - (start,left))
- (start,right) = (start,right) + without_advise_lr * (without_advise_opposite_reward - (start,right))

#### 1.2. Forget
- No forgeting rule is applied to (advise_left,right)

### 2. Disconneted Version
#### 2.1. Update
- (advise_left,right) = (advise_left,right) + lr * (actual_reward - (advise_left,right))
- (advise_right,left) = (advise_right,left) + lr * (actual_reward - (advise_right,left))
- (start,right) = (start,right) + without_advise_lr * (without_advise_actual_reward - (start,right))
#### 2.2. Forget
- (advise_right,right) = (advise_right,right) + fr * (advise_right_right_intit_vale - (advise_right,right))
- (advise_left,left) = (advise_left,left) + fr * (advise_left_left_intit_vale - (advise_left,left))
- (start,left) = (start,left) + without_advise_fr * (start_left_intit_vale - (start,left))

## (advise_right,left)
### 1. Conneted Version
#### 1.1. Update
- (advise_right,left) = (advise_right,left) + lr * (actual_reward - (advise_right,left))
- (advise_left,right) = (advise_left,right) + lr * (actual_reward - (advise_left,right))
- (advise_left,left) = (advise_left,left) + lr * (opposite_reward - (advise_left,left))
- (advise_right,right) = (advise_right,right) + lr * (opposite_reward - (advise_right,right))
- (start,left) = (start,left) + without_advise_lr * (without_advise_actual_reward - (start,left))
- (start,right) = (start,right) + without_advise_lr * (without_advise_opposite_reward - (start,right))

#### 1.2. Forget
- No forgeting rule is applied to (advise_right,left)

### 2. Disconneted Version
#### 2.1. Update
- (advise_right,left) = (advise_right,left) + lr * (actual_reward - (advise_right,left))
- (advise_left,right) = (advise_left,right) + lr * (actual_reward - (advise_left,right))
- (start,left) = (start,left) + without_advise_lr * (without_advise_actual_reward - (start,left))
#### 2.2. Forget
- (advise_left,left) = (advise_left,left) + fr * (advise_left_left_intit_vale - (advise_left,left))
- (advise_right,right) = (advise_right,right) + fr * (advise_right_right_intit_vale - (advise_right,right))
- (start,right) = (start,right) + without_advise_fr * (start_right_intit_vale - (start,right))


## (advise_right,right)
### 1. Conneted Version
#### 1.1. Update
- (advise_right,right) = (advise_right,right) + lr * (actual_reward - (advise_right,right))
- (advise_left,left) = (advise_left,left) + lr * (actual_reward - (advise_left,left))
- (advise_left,right) = (advise_left,right) + lr * (opposite_reward - (advise_left,right))
- (advise_right,left) = (advise_right,left) + lr * (opposite_reward - (advise_right,left))
- (start,left) = (start,left) + without_advise_lr * (without_advise_opposite_reward - (start,left))
- (start,right) = (start,right) + without_advise_lr * (without_advise_actual_reward - (start,right))

#### 1.2. Forget
- No forgeting rule is applied to (advise_right,right)


### 2. Disconneted Version
#### 2.1. Update
- (advise_right,right) = (advise_right,right) + lr * (actual_reward - (advise_right,right))
- (advise_left,left) = (advise_left,left) + lr * (actual_reward - (advise_left,left))
- (start,right) = (start,right) + without_advise_lr * (without_advise_actual_reward - (start,right))
#### 2.2. Forget
- (advise_left,right) = (advise_left,right) + fr * (advise_left_right_intit_vale - (advise_left,right))
- (advise_right,left) = (advise_right,left) + fr * (advise_right_left_intit_vale - (advise_right,left))
- (start,left) = (start,left) + without_advise_fr * (start_left_intit_vale - (start,left))



