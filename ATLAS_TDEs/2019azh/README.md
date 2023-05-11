# SN 2019azh Light Curve Cleaning and Averaging

The ATLAS SN light curves are separated by filter (orange and cyan) and labelled as such in the file name. Averaged light curves contain an additional number in the file name that represents the MJD bin size used. Control light curves are located in the "controls" subdirectory and follow the same naming scheme, only with their control index added after the SN name.

The following details the file names for each of the light curve versions:
	- SN light curves: 2019azh.o.lc.txt and 2019azh.c.lc.txt
	- Averaged light curves: 2019azh.o.1.00days.lc.txt and 2019azh.c.1.00days.lc.txt

The following summarizes the hex values in the "Mask" column of each light curve for each cut applied (see below sections for more information on each cut): 
	- Uncertainty cut: 0x2
	- Bad day (for averaged light curves): 0x800000

## FILTER: o

### Correction for ATLAS reference template changes
We take into account ATLAS's periodic replacement of the difference image reference templates, which may cause step discontinuities in flux. Two template changes have been recorded at MJDs 58417 and 58882.
Correction applied to baseline region 0: 0.0 uJy subtracted
Correction applied to baseline region 1: 0.0 uJy subtracted
Correction applied to baseline region 2: 0.0 uJy subtracted

### Uncertainty cut
We flag measurements with an uncertainty (column name "duJy") value above 60.00 with hex value 0x2.

Total percent of data flagged: 8.68%.

### Estimating true uncertainties
This procedure attempts to account for an extra noise source in the data by estimating the true typical uncertainty, deriving the additional systematic uncertainty, and lastly applying this extra noise to a new uncertainty column "duJy_new". This new uncertainty column will be used in the cuts following this portion. An extra noise of sigma 111.8897 was added to the uncertainties of the SN light curve and copied to the "duJy_new" column.

### After the uncertainty, chi-square, and control light curve cuts are applied, the light curves are resaved with the new "Mask" column.

### Averaging light curves and cutting bad days
For each MJD bin of size 1.00 day(s), we calculate the 3σ-clipped average of any SN measurements falling within that bin and use that average as our flux for that bin. However, out of all exposures within this MJD bin, only measurements not cut in the previous methods are averaged in the 3σ-clipped average cut. (The exception to this statement would be the case that all 4 measurements are cut in previous methods; in this case, they are averaged anyway and flagged as a bad bin.

Then we flag any measurements in the SN light curve for the given epoch for which statistics fulfill any of the following criteria with the hex value 0x800000: 
	- A returned chi-square > 4.0
	- Number of measurements averaged < 2
	- Number of measurements clipped > 1

The averaged light curves are then saved in a new file with the MJD bin size added to the filename.

Total percent of data flagged: 4.36%.

## FILTER: c

### Correction for ATLAS reference template changes
We take into account ATLAS's periodic replacement of the difference image reference templates, which may cause step discontinuities in flux. Two template changes have been recorded at MJDs 58417 and 58882.
Correction applied to baseline region 0: 0.0 uJy subtracted
Correction applied to baseline region 1: 0.0 uJy subtracted
Correction applied to baseline region 2: 0.0 uJy subtracted

### Uncertainty cut
We flag measurements with an uncertainty (column name "duJy") value above 60.00 with hex value 0x2.

Total percent of data flagged: 4.96%.

### Estimating true uncertainties
This procedure attempts to account for an extra noise source in the data by estimating the true typical uncertainty, deriving the additional systematic uncertainty, and lastly applying this extra noise to a new uncertainty column "duJy_new". This new uncertainty column will be used in the cuts following this portion. An extra noise of sigma 71.4207 was added to the uncertainties of the SN light curve and copied to the "duJy_new" column.

### After the uncertainty, chi-square, and control light curve cuts are applied, the light curves are resaved with the new "Mask" column.

### Averaging light curves and cutting bad days
For each MJD bin of size 1.00 day(s), we calculate the 3σ-clipped average of any SN measurements falling within that bin and use that average as our flux for that bin. However, out of all exposures within this MJD bin, only measurements not cut in the previous methods are averaged in the 3σ-clipped average cut. (The exception to this statement would be the case that all 4 measurements are cut in previous methods; in this case, they are averaged anyway and flagged as a bad bin.

Then we flag any measurements in the SN light curve for the given epoch for which statistics fulfill any of the following criteria with the hex value 0x800000: 
	- A returned chi-square > 4.0
	- Number of measurements averaged < 2
	- Number of measurements clipped > 1

The averaged light curves are then saved in a new file with the MJD bin size added to the filename.

Total percent of data flagged: 1.46%.