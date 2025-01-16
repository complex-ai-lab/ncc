# Download power consumption dataset
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip -O ./data/household_power_consumption.zip
mkdir -p ./data/household_power_consumption
unzip ./data/household_power_consumption.zip -d ./data/household_power_consumption
rm ./data/household_power_consumption.zip