import pandas as pd
import random

# create a dataset for rainfall and flood, Random production 
def generate_flood_dataset(numberofregions):
    JAN = []
    FEB = []
    MAR	= []
    APR	= []
    MAY	= []
    JUN	= []
    JUL	= []
    AUG	= []
    SEP	= []
    OCT	= []
    NOV	= []
    DEC = []
    FLOODS = []
    REGION = []
    for r in range(numberofregions):
        REGION.append(str(f'region_{r}'))
        # Rainfall for each region
        JAN.append(float(f'{random.uniform(600,1200):.2f}'))
        FEB.append(float(f'{random.uniform(700,1200):.2f}'))
        MAR.append(float(f'{random.uniform(450,750):.2f}'))
        APR.append(float(f'{random.uniform(315,600):.2f}'))
        MAY.append(float(f'{random.uniform(70,240):.2f}'))
        JUN.append(float(f'{random.uniform(20,80):.2f}'))
        JUL.append(float(f'{random.uniform(1.2,60):.2f}'))
        AUG.append(float(f'{random.uniform(0.2,35):.2f}'))
        SEP.append(float(f'{random.uniform(120,500):.2f}'))
        OCT.append(float(f'{random.uniform(130,700):.2f}'))
        NOV.append(float(f'{random.uniform(230,800):.2f}'))
        DEC.append(float(f'{random.uniform(500,1000):.2f}'))
    
    df = pd.DataFrame({'REGION':REGION,'JAN':JAN,'FEB':FEB,'MAR':MAR,
                       'APR':APR,'MAY':MAY,'JUN':JUN,'JUL':JUL,'AUG':AUG,
                       'SEP':SEP,'OCT':OCT,'NOV':NOV,'DEC':DEC})
    # -------
    # Floods
    St_D = df.std(axis=1,numeric_only=True)
    th = (max(St_D)+min(St_D))/2
    for i in range(numberofregions):
        if St_D[i] < th:
            FLOODS.append('NO')
        else: 
            FLOODS.append('YES')
    df['FLOODS'] = FLOODS

    # -------
    # Column for the annual rainfall
    ANNUAL_RAINFALL = list(df.JAN + df.FEB + df.MAR + df.APR + df.MAY + df.JUN + 
                           df.JUL + df.AUG + df.SEP + df.OCT + df.NOV + df.DEC)
    df['ANNUAL_RAINFALL'] = ANNUAL_RAINFALL

    # -------
    # Save dataset in a .csv file
    df.to_csv('floods.csv', index=False) 

generate_flood_dataset(1000)