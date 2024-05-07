from astropy import units as u

SI = u.S/u.m
new = u.mS/u.cm

print(SI.to(new, 1))
