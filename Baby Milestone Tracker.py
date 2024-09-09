import matplotlib.pyplot as plt
import pandas as pd 
milestones = {'Date':['2023-12-21', '2023-01-05', '2024-03-22', '2024-05-31', '2024-06-21' ],
             'Milestone': ['Born', 'first smile', 'First haircut','Teething', 'Sitting up'],
              'Description': ['Arrived at 6.30 AM', 'First smile in his sleep captured on Camera',
                              'Got his first haircut from Shosh', 'Started wanting to chew on hard things',
                              'Sat up for the first time captured on Camera'], 
              'color': ['blue', 'green', 'red', 'orange', 'purple']
                }
#Convert into a dataframe
df = pd.DataFrame(milestones)
#Converting date strings to datetime objects
df['Date'] = pd.to_datetime(df['Date'])
#Plotting the milestones
plt.figure(figsize = (10,5))
plt.scatter(df['Date'], [1]*len(df), c=df['Color'], s=200, zorder=2)
#Drawing the timeline
plt.plot([df['Date'].min(), df['Date'].max()], [1, 1], color='gray', lw=2, zorder=1)
#Annotating the milestones
for i, row in df.iterrows():plt.text(row['Date'], 1.05, row['Milestone'], ha='center',fontsize=10, 
                                     color=row['color'])
plt.text(row['Date'], 0.95, row['Description'], ha='center', fontsize=8, color='black', alpha=0.7)
#Formatting the plot
plt.yticks([])
plt.xticks(rotation=45)
plt.title("Arthur's First Year Milestone Tracker")
plt.grid(False)
plt.tight_layout()
plt.show()
