import  argparse

def Args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cover_path', default='../srnet_data/training/cover/')
	parser.add_argument('--stego_path', default='../srnet_data/training/suni04R/')
	parser.add_argument('--valid_cover_path', default='../srnet_data/val/cover/')
	parser.add_argument('--valid_stego_path', default='../srnet_data/val/suni04R/')
	parser.add_argument('--checkpoints_dir', default='./checkpoints/')
	parser.add_argument('--batch_size', type=int, default=20)
	parser.add_argument('--num_epochs', type=int, default=500)
	parser.add_argument('--train_size', type=int, default=14000)
	parser.add_argument('--val_size', type=int, default=1000)
	parser.add_argument('--p_rot', type=float, default=0.1)
	parser.add_argument('--p_hflip', type=float, default=0.1)
	parser.add_argument('--lr', type=float, default=0.001)

	opt = parser.parse_args()
	return opt