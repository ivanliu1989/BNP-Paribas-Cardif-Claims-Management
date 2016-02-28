# unvectorized version

az_to_int <- function(az) {
  xx <- strsplit(tolower(az), "")[[1]]
  pos <- match(xx, letters[(1:26)]) 
  result <- sum( pos* 26^rev(seq_along(xx)-1))
  return(result)
}

train$v22<-sapply(train$v22, az_to_int)

# Or even simpler (in R again):
levels <- unique(train_test$v22)
levels <- levels[order(nchar(levels), tolower(levels))]
train_test$v22 <- as.integer(factor(train_test$v22, levels=levels))

# python
all_data.v22.fillna('',inplace=True)

# Padding v22 to 4 characters allows sorting to work correctly
padded = all_data.v22.str.pad(4)
spadded = sorted(np.unique(padded))

# Map sorted v22 values so they wind up in the best order
v22_map = {}
c = 0
for i in spadded:
    v22_map[i] = c
    c += 1

# Now apply to v22
all_data.v22 = padded.replace(v22_map, inplace=False)