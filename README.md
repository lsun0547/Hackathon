# Hackathon

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
- Hour 0 to hour 24

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