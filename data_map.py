import os
datain = './data'
dataout = './result'

if not os.path.exists(dataout):
    os.makedirs(dataout)

feature = ['Age','Sex','Job','Housing','Saving accounts','Checking account','Credit amount','Duration','Purpose']
label = ['Risk']

sex = {'male' : 0, 'female' : 1}
housing = {'free' : 0, 'rent' : 1, 'own' : 2}
saving = {'little' : 0, 'moderate' : 1, 'rich' : 2, 'quite rich' : 3}
checking = {'little' : 0, 'moderate' : 1, 'rich' : 2}
purpose = \
    {'radio/TV' : 0,
     'education' : 1,
     'furniture/equipment' : 2,
     'car' : 3,
     'business' : 4,
     'domestic appliances' : 5,
     'repairs' : 6,
     'vacation/others' : 7}
risk = {'bad' : 0, 'good' : 1}
