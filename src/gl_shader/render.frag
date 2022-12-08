#version 330 core
in vec2 texCoord;
out vec4 FragColor;
uniform sampler2D boundary_texture;	//boundary texture map specifying boundaries
uniform sampler2D state_texture3;	    //input texture containing f0, rho, ux and uy

void main()
{

    vec2 pos = texCoord.xy;		//	Position of each lattice node	

    if ( texture2D( boundary_texture,pos ).x > 0.5 ) {
        float color = texture2D( state_texture3, pos ).y;
        FragColor = vec4( color * 0.4, color * 0.6, color, 0.0 );
    } else {
        // boundary, draw black
        FragColor = vec4(0.0, 0.0, 0.0, 0.0);
    }
    
}