subroutine pysiestaforce ( label, na, xa, cell, energy, fa )
  
  use fsiesta

  implicit none
!  integer, parameter    ::  dp = kind(1.d0)
  integer, parameter    ::  dp = 8

  character(*),          intent(in) :: label
  integer,            intent(in) :: na
  real(dp),           intent(in) :: xa(3*na)
  real(dp),           intent(in) :: cell(3*3)
  real(dp),           intent(out):: energy
  real(dp),           intent(out):: fa(3*na)

  integer           :: i,j
  real(dp)          :: ffa(3,na)
  real(dp)          :: ccell(3,3)
  real(dp)          :: xxa(3,na)

  do j=1,na
    do i=1,3
      xxa(i,j) = xa(3*(j-1)+i)
    enddo
  enddo
  
  do j=1,3
    do i=1,3
      ccell(i,j)=cell(3*(j-1)+i)
    enddo
  enddo


  call siesta_forces( label, na, xxa, ccell, energy, ffa )


  do j=1,na
    do i=1,3
      fa(3*(j-1)+i) = ffa(i,j) 
    enddo
  enddo

end subroutine pysiestaforce

subroutine pysiestalaunch( label, nn, command)
  use fsiesta

  implicit none

  character(*),          intent(in) :: label
  integer, optional,           intent(in) :: nn
  character(*),optional,       intent(in) :: command

  
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !This works for siesta-3.0
  !if(present(nn).and.present(command))then
  !    !call siesta_launch( label, nnodes=nnodes, mpi_launcher=command )
  !    call siesta_launch( label, nnodes=nnodes, launcher=command )
  !elseif(.not. present(nn) .or. .not. present(command))then
  !    call siesta_launch( label )
  !endif
  !This works for siesta-3.0
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !This works for siesta-4.1.5
  if(present(command))then
      !call siesta_launch( label, nnodes=nn, mpi_launcher=command )
      call siesta_launch( label, launcher=command )
  elseif(present(nn))then
      print*, 'nnodes=', nn
      call siesta_launch( label, nnodes=nn)
  else
      print*, 'no nnodes, run serial'
      call siesta_launch( label )
  endif
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  return
end subroutine pysiestalaunch

subroutine pysiestaquit( label )
  use fsiesta

  implicit none

  character(*),          intent(in) :: label
  
  print*, label
  call siesta_quit ( label )
  print*, 'siesta quit'
  return
end subroutine pysiestaquit

subroutine pysiestaunits( length, energy )
  use fsiesta

  implicit none

  character(*),          intent(in) :: length
  character(*),          intent(in) :: energy
  
  call siesta_units( length, energy )
  print*, 'units set to:'
  print*, 'length:', length
  print*, 'energy:', energy
  return
end subroutine pysiestaunits
