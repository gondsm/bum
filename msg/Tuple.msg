# The ID of the characteristic at hand
string char_id

# The classification value
int32 characteristic

# The evidence used to determine the characteristic, corresponding to this
# datapoint. It must be in the order specified in the GDC! (listeners 
# assume this).
int32[] evidence

# The ID of the active user
uint32 user_id

# The entropy or certainty level of the classification
float32 h