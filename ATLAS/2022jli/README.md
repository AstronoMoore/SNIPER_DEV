# SN 2022jli Light Curve Cleaning and Averaging

The ATLAS SN light curves are separated by filter (orange and cyan) and labelled as such in the file name. Averaged light curves contain an additional number in the file name that represents the MJD bin size used. Control light curves are located in the "controls" subdirectory and follow the same naming scheme, only with their control index added after the SN name.

The following details the file names for each of the light curve versions:
	- SN light curves: 2022jli.o.lc.txt and 2022jli.c.lc.txt
	- Averaged light curves: 2022jli.o.1.00days.lc.txt and 2022jli.c.1.00days.lc.txt

The following summarizes the hex values in the "Mask" column of each light curve for each cut applied (see below sections for more information on each cut): 
	- Uncertainty cut: 0x2
	- Chi-square cut: 0x1
	- Bad day (for averaged light curves): 0x800000

## FILTER: o

### Uncertainty cut
We flag measurements with an uncertainty (column name "duJy") value above 160.00 with hex value 0x2.

Total percent of data flagged: 3.73%.

### Estimating true uncertainties
This procedure attempts to account for an extra noise source in the data by estimating the true typical uncertainty, deriving the additional systematic uncertainty, and lastly applying this extra noise to a new uncertainty column. This new uncertainty column will be used in the cuts following this portion. An extra noise of sigma 23.7818 was added to the uncertainties.

### Chi-square cut
We flag measurements with a chi-square (column name "chi/N") value above 3.00 with hex value 0x1.
	- The cut optimized according to the given contamination limit of 15.00% was 160.00, with a contamination of 0.25% and a loss of 0.00%.
	- The cut optimized according to the given loss limit of 10.00% was 3.00, with a contamination of 0.25% and a loss of 0.00%.

Total percent of data flagged: 6.70%.

### After the uncertainty, chi-square, and control light curve cuts are applied, the light curves are resaved with the new "Mask" column.

### Averaging light curves and cutting bad days
For each MJD bin of size 1.00 day(s), we calculate the 3σ-clipped average of any SN measurements falling within that bin and use that average as our flux for that bin. However, out of all exposures within this MJD bin, only measurements not cut in the previous methods are averaged in the 3σ-clipped average cut. (The exception to this statement would be the case that all 4 measurements are cut in previous methods; in this case, they are averaged anyway and flagged as a bad bin.

Then we flag any measurements in the SN light curve for the given epoch for which statistics fulfill any of the following criteria with the hex value 0x800000: 
	- A returned chi-square > 4.0
	- Number of measurements averaged < 2
	- Number of measurements clipped > 1

The averaged light curves are then saved in a new file with the MJD bin size added to the filename.

Total percent of data flagged: 17.23%.

## FILTER: c

### Uncertainty cut
We flag measurements with an uncertainty (column name "duJy") value above 160.00 with hex value 0x2.

Total percent of data flagged: 1.96%.

### Estimating true uncertainties
This procedure attempts to account for an extra noise source in the data by estimating the true typical uncertainty, deriving the additional systematic uncertainty, and lastly applying this extra noise to a new uncertainty column. This new uncertainty column will be used in the cuts following this portion. An extra noise of sigma 16.7711 was added to the uncertainties.

### Chi-square cut
We flag measurements with a chi-square (column name "chi/N") value above 3.00 with hex value 0x1.
	- The cut optimized according to the given contamination limit of 15.00% was 160.00, with a contamination of 0.00% and a loss of 0.00%.
	- The cut optimized according to the given loss limit of 10.00% was 3.00, with a contamination of 0.00% and a loss of 0.63%.

Total percent of data flagged: 12.85%.

### After the uncertainty, chi-square, and control light curve cuts are applied, the light curves are resaved with the new "Mask" column.

### Averaging light curves and cutting bad days
For each MJD bin of size 1.00 day(s), we calculate the 3σ-clipped average of any SN measurements falling within that bin and use that average as our flux for that bin. However, out of all exposures within this MJD bin, only measurements not cut in the previous methods are averaged in the 3σ-clipped average cut. (The exception to this statement would be the case that all 4 measurements are cut in previous methods; in this case, they are averaged anyway and flagged as a bad bin.

Then we flag any measurements in the SN light curve for the given epoch for which statistics fulfill any of the following criteria with the hex value 0x800000: 
	- A returned chi-square > 4.0
	- Number of measurements averaged < 2
	- Number of measurements clipped > 1

The averaged light curves are then saved in a new file with the MJD bin size added to the filename.

Total percent of data flagged: 30.09%.