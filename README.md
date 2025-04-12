# Hackathon
- The theme of the hackathon was to devise a creative solution to solve a peaceful graveyard's unpredictable and mysterious hauntings. We created a fake dataset of hauntings in the graveyard. Then, we implemented both a logistic regression mode and a random forest model to predict whether a person would get haunted or not, given the variables.
- After this, we imported this model into our HTML + CSS front end, allowing users to provide the conditions they are in. This will predict whether they will be haunted or not.
- There are a few potential uses for this. It was mentioned that the cause of the haunting was unknown. With more knowledge regarding the circumstances behind the hauntings, we believe it could be solved more easily. Additionally, it could be used to allow the villagers to avoid ghosts on days the graveyard is more likely to be haunted.
---

# GhostDataSetTest.py

## Import Statements
- **Pandas** for our data frame
- **Faker** for our dataset

---

## Creating Variables

### Location Options:
- North Gate
- Old Oak Tree
- Mausoleum
- Grave 42
- Chapel
- Back Fence

### Weather Options:
- Clear
- Foggy
- Rainy
- Stormy
- Windy

### Moon Options:
- New Moon
- Young Moon
- Waxing Crescent
- Waxing Quarter
- Waxing Gibbous
- Full Moon
- Waning Gibbous
- Waning Quarter
- Waning Crescent

### Temperature Options:
- Very Cold
- Cold
- Mild
- Warm
- Hot

### Time Options:
- Hour 0 to hour 23

### Day Options:
- Monday
- Tuesday
- Wednesday
- Thursday
- Friday
- Saturday
- Sunday

### Wind Options:
- None
- Light Wind
- Steady Wind
- Strong Wind
- Howling Wind

---

## Function: `generate_entry()`

- For each variable, generate a random option.
  - The **moon** variable can be a *full moon* or a *new moon*.
  - The **day** variable can be any day of the week.

### Conditions:
1. **Haunting Chance: 60%**
   - If the weather is foggy or stormy
   - OR if the user is in the Mausoleum or Chapel

2. **Haunting Chance: 30%**
   - If the moon is a full moon
   - OR if the user is near a Grave

3. **Haunting Chance: 20%**
   - For all other conditions

4. **Return**
   - Return the various variables we have

## Dataset

- Generate the dataset using pandas
  - Generate 5000 entries
  - Create the dataframe and name it df

- Make the df a .csv file which can be saved into PyCharm

---

# GhostAnalysis.py
## Import Statements

- **Seaborn** for machine learning
- **Pandas** and **Matplotlib** for data analysis
- Various **Sci-kit Learn** libraries will also be utilized

## Dataframes
- Assign each variable a different index in the dataframe
- Create the haunted and not haunted options (1 and 0, respectively)
- Balance the dataset with sampled and unsampled parts
- Split the data
  - Create a test and train dataset
  - 80% will be dedicated for train, 20% for test
- Prediction
  - **Random Forest Generator** was our model of choice
  - Calculate our accuracy
  - Print out our prediction and the ground truth
