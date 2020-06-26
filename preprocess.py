import numpy as np
import csv
import copy


# Gathering Data
print("Gathering and Preprocessing Data")
Data = []
with open("/Users/hridaybhambhvani/UroRhabdo/uro.csv") as file:
	CSV = csv.reader(file, delimiter=',')
	for row in CSV:
		Data.append(row)
Data = np.array(Data)

# Separate Headers and Patient IDs from Data
Headers = Data[0,1:]
Data = Data[1:]
IDs = Data[:,0]
Data = Data[:,1:]
print(Headers, Data[0])


clean = True
for i in range(len(Headers)):
	if i == 0:
		print("Preprocessing", Headers[i])

		Age = Data[:,i].astype(float)
		print("Preprocessed Age")

	if i == 1:
		print("Preprocessing", Headers[i])
		# Unknown, White, Black, Asian/PI, AI/AN --> 0, 1, 2, 3, 4
		Race = Data[:,i]
		white = Race == 'White'
		black = Race == 'Black'
		as_pi = Race == 'Asian or Pacific Islander'
		ai_an = Race == 'American Indian/Alaska Native'
		unkwn = Race == 'Unknown'

		# Recode
		Race[unkwn] = '-1'
		Race[white] = '1'
		Race[black] = '2'
		Race[as_pi] = '3'
		Race[ai_an] = '4'

		# Assertion
		All = [unkwn, white, black, as_pi, ai_an]
		All = np.sum([a.sum() for a in All])
		assert(All == len(Race))

		print(f'Race : {unkwn.sum()} Unknown, {white.sum()} White, {black.sum()} Black, {as_pi.sum()} AS/PI, {ai_an.sum()} AI/AN')

		race_unkwn = copy.deepcopy(unkwn)

		Race = Race.astype(float)
		print("Preprocessed Race")

	if i == 2:
		print("Preprocessing", Headers[i])
		# Male, Female --> 0, 1
		Sex = Data[:,i]
		M = Sex == 'Male'
		F = Sex == 'Female'
		
		Sex[M] = '0'
		Sex[F] = '1'

		# Assertion
		All = [M, F]
		All = np.sum([a.sum() for a in All])
		assert(All == len(Sex))

		print(f'Sex : {M.sum()} Male, {F.sum()} Female')
		Sex = Sex.astype(float)
		print("Preprocessed Sex")

	if i == 3:
		print("Preprocessing", Headers[i])
		# Vagina, Prostate, Urinary Bladder, Testis, Cervix Uteri, Ovary, Vulva, Uterus NOS, Other MGO, Other -> 0, 1, 2, 3, 4, 5, 6, 7, 9, 10 (no 8 for space + in case I missed one)
		Site = Data[:,i]

		vagina = Site == 'Vagina'
		prostate = Site == 'Prostate'
		ub = Site == 'Urinary Bladder'
		testis = Site == 'Testis'
		cu = Site == 'Cervix Uteri'
		ovary = Site == 'Ovary'
		vulva = Site == 'Vulva'
		unos = Site == 'Uterus, NOS'
		omgo = Site == 'Other Male Genital Organs'
		other = Site == 'Other'

		Site[vagina] = '0'
		Site[prostate] = '1'
		Site[ub] = '2'
		Site[testis] = '3'
		Site[cu] = '4'
		Site[ovary] = '5'
		Site[vulva] = '6'
		Site[unos] = '7'

		Site[omgo] = '9' 
		Site[other] = '10'

		# Assertion
		All = [vagina, prostate, ub, testis, cu, ovary, vulva, unos, omgo, other] 
		All = np.sum([a.sum() for a in All])
		assert(All == len(Site))

		'''
		if clean:
			Site = Site[np.invert(unkwn)]
		else: 
			site_unkwn = copy.deepcopy(unkwn)
		'''

		print(f'Site : {vagina.sum()} Vagina, {prostate.sum()} Prostate, {ub.sum()} Urinary Bladder, {testis.sum()} Testis, {cu.sum()} Cervix Uteri, {ovary.sum()} Ovary, {vulva.sum()} Vulva, {unos.sum()} Uterus NOS, {omgo.sum()} OMGO, {other.sum()} Other')

		Site = Site.astype(float)
		print("Preprocessed Site")

	if i == 4:
		print("Preprocessing", Headers[i])
		Surv = Data[:,i].astype(float)
		Vital = Data[:,i+1]
		Alive = Vital == 'Alive'
		Dead = Vital == 'Dead'
		Cause = Data[:,i+2]
		DeadDX = Cause == 'Dead (attributable to this cancer dx)'

		Y = Surv >= 60
		sp_N = np.logical_and(Surv < 60, DeadDX)
		ov_N = np.logical_and(Surv < 60, Dead)

		print(f'Vitality : {Alive.sum()} Alive, {Dead.sum()} Dead, {DeadDX.sum()} DeadDX')

		usable = np.invert(np.logical_and(Surv < 60, Alive))
		print("Preprocessed Survival (Labels)")

	if i == 7:
		print("Preprocessing", Headers[i])
		Stage = Data[:,i]

		unstaged = Stage == 'Unstaged'
		localized = Stage == 'Localized'
		regional = Stage == 'Regional'
		distant = Stage == 'Distant'

		Stage[unstaged] = '0'
		Stage[localized] = '1'
		Stage[regional] = '2'
		Stage[distant] = '3'

		All = [unstaged, localized, regional, distant] 
		All = np.sum([a.sum() for a in All])
		assert(All == len(Stage))

		print(f'Stage : {unstaged.sum()} Unstaged, {localized.sum()} Localized, {regional.sum()} Regional, {distant.sum()} Distant')
		Stage = Stage.astype(float)
		print("Preprocessed Stage")


	if i == 8:
		print("Preprocessing", Headers[i])

		Size = Data[:,i]

		unkwn = Size == '999'

		size_unkwn = copy.deepcopy(unkwn)

		Size = Size.astype(float)/100.
		print("Preprocessed Size")

	if i == 9:
		print("Preprocessing", Headers[i])

		Surg = Data[:,i].astype(float)
		print("Preprocessed Surg")

	if i == 10:
		print("Preprocessing", Headers[i])

		Rad = Data[:,i].astype(float)
		print("Preprocessed Rad")

	if i == 11:
		print("Preprocessing", Headers[i])
		Hist = Data[:,i]

		Emb = Hist == 'Embryonal'
		Alv = Hist == 'Alveolar'
		oth = Hist == 'Other'
		unkwn = Hist == 'Unknown'

		Hist[Emb] = '0'
		Hist[Alv] = '1'
		Hist[oth] = '2'
		Hist[unkwn] = '5'

		All = [Emb, Alv, oth, unkwn] 
		All = np.sum([a.sum() for a in All])
		assert(All == len(Hist))

		hist_unkwn = copy.deepcopy(unkwn)

		print(f'Hist : {Emb.sum()} Embryonal, {Alv.sum()} Alveolar, {oth.sum()} Other, {unkwn.sum()} Unknown')
		Hist = Data[:,i].astype(float)
		print("Preprocessed Hist")

Data = np.array([Age, Race, Sex, Site, Stage, Size, Surg, Rad, Hist]).T
#Labels = np.array([Surv, Surv>=60]).T
spec = np.array([Surv, np.logical_or(Surv >= 60, np.logical_and(Surv < 60, Cause == 'Alive or dead of other cause'))]).T # Cancer Specific
over = np.array([Surv, Surv >= 60]).T # Overall

if clean:
	usable = np.logical_and(np.invert(np.logical_or(np.logical_or(race_unkwn, size_unkwn), hist_unkwn)), usable)
Data = Data[usable]
#Labels = Labels[usable]
spec = spec[usable]
over = over[usable]
IDs = IDs[usable]

print(f'Specific = {spec[:,1].astype(int).sum()}, Overall = {over[:,1].astype(int).sum()}')

print(Data.shape)
np.savez('Data.npz', Data=Data, spec=spec, over=over, IDs=IDs)















