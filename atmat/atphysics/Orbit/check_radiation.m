function [radon, ring] = check_radiation(ring, onoff, varargin)
%CHECK_RADIATION	Check the radiation state of a lattice
%
%RADON = CHECK_RADIATION(RING)
%   Return the radiation state of RING
%
%RADON = CHECK_RADIATION(RING,DESIRED)
%   Throw an error if the radiation is not DESIRED
%
%[RADON, NEWRING] = CHECK_RADIATION(RING, DESIRED, 'force')
%   

[force, varargs] = getflag(varargin, 'force'); %#ok<ASGLU>

radon=false;
for i=1:length(ring)
    passmethod=ring{i}.PassMethod;
    if endsWith(passmethod, {'RadPass', 'CavityPass', 'QuantDiffPass'})
        radon=true;
        break;
    end
end

if nargin >= 2
    if xor(radon,onoff)
        if force
            if onoff
                ring = atradon(ring);
            else
                ring = atradoff(ring);
            end
        else
            error('AT:Radiation',['Radiation must be ' boolstring(onoff)]);
        end
    end
end

    function bstr=boolstring(onoff)
        if onoff
            bstr='ON';
        else
            bstr='OFF';
        end
    end
end
