from ucimlrepo import fetch_ucirepo 

print('hayes_roth')
# fetch dataset 
hayes_roth = fetch_ucirepo(id=44) 
  
# data (as pandas dataframes) 
X = hayes_roth.data.features 
y = hayes_roth.data.targets 
  
# metadata 
print(hayes_roth.metadata) 
  
# variable information 
print(hayes_roth.variables) 


  
