function radon=check_radiation(ring, onoff)
%CHECK_RADIATION	Throw error is the radiation state is not the expected one

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
        error('AT:Radiation',['Radiation must be ' boolstring(onoff)])
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
