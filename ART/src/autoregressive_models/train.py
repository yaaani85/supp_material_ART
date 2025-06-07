
from src.config import get_cli_args
from src.engine import Engine

def main():
    args = get_cli_args()
    config = vars(args)
    engine = Engine(config)
    engine.train()

if __name__ == '__main__':
    main()