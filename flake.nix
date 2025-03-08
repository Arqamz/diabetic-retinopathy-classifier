{ description = "Dev shell for Python DL";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }: let
      pkgs = nixpkgs.legacyPackages."x86_64-linux";
  in {
    devShells.x86_64-linux = {
      default = pkgs.mkShell {
      
        packages = [
  	  (pkgs.python312.withPackages(pypkgs: with pypkgs; [
             pip
	  ]))     
        ];

	env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
	  pkgs.stdenv.cc.cc.lib
	  pkgs.libz
	];

        shellHook = ''
          if [ ! -d ".venv" ]; then
            python3 -m venv .venv
            echo "Virtual environment created."
          else
            echo "Virtual environment already exists."
          fi
     
          source .venv/bin/activate

          echo "Development environment for Deep Learning with Python is ready."
          echo "Python version: $(python3 --version)"
        '';
      };
    };
  };
}
